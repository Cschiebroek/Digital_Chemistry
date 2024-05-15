hp_search_config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([8, 16]),
        "hidden_channels": tune.choice([32, 64, 128]),
        "num_layers": tune.choice([2, 3, 4]),
        "num_timesteps": tune.choice([1, 2, 3]),
        "gamma": tune.loguniform(0.9, 0.99),
        "Scheduler": tune.choice(["ReduceLROnPlateau", "ExponentialLR"]),
    }

def tune_hyperparameters(self,config, num_samples=10, max_num_epochs=50, gpus_per_trial=0):
        """
        Tune hyperparameters for the model using Ray Tune with the ASHA scheduler.

        Args:
            data_train: Training dataset.
            data_val: Validation dataset.
            num_samples (int): Number of trials to run. Defaults to 10.
            max_num_epochs (int): Maximum number of epochs for training. Defaults to 50.
            gpus_per_trial (float): Number of GPUs to allocate per trial. Defaults to 0.

        """
        def compute_weighted_loss(targets, predictions, outputs):
            """
            Computes the weighted loss for the model.

            Args:
                targets (Tensor): The target values.
                predictions (Tensor): The predicted values.
                outputs (int): The number of outputs of the model.

            Returns:
                Tuple: A tuple containing the weighted loss and the number of labels.
            """
            weighted_loss = num_labels = 0
            for i in range(outputs):
                y_tmp = targets[:, i]
                out_tmp = predictions[:, i]
                present_label_indices = torch.nonzero(y_tmp != 0, as_tuple=False).view(-1)
                num_labels += len(present_label_indices)

                if len(present_label_indices) > 0:
                    out_tmp_present = out_tmp[present_label_indices]
                    y_tmp_present = y_tmp[present_label_indices]
                    loss_tmp_present = F.mse_loss(out_tmp_present, y_tmp_present)
                    weighted_loss += loss_tmp_present * len(present_label_indices)

            return weighted_loss / num_labels if num_labels > 0 else 0, num_labels
        def train_step(data, model, optimizer, device,config,outputs):
            """
            Trains the model for the specified number of epochs.

            Args:
                train_loader (DataLoader): The data loader for the training data.
                epochs (int): The number of epochs to train the model.
                outputs (int): The number of outputs of the model.
            """

            model.train()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=True)

            for data in loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                weighted_loss, num_labels = compute_weighted_loss(data.y, out, outputs)
                weighted_loss.backward()
                total_loss += float(weighted_loss) * data.num_graphs
                total_examples += data.num_graphs
                optimizer.step()
            return np.sqrt(total_loss / total_examples)


        def validate(data, model, device,config,outputs):
            """
            Validates the model using the validation data.

            Args:
                val_loader (DataLoader): The data loader for the validation data.
                outputs (int): The number of outputs of the model.
            """
            model.eval()
            model.to(device)
            total_loss = total_examples = 0
            data = ray.get(data)
            loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    weighted_loss, num_labels = compute_weighted_loss(data.y, out, outputs)
                    total_loss += float(weighted_loss) * data.num_graphs
                    total_examples += data.num_graphs

            return np.sqrt(total_loss / total_examples)

        
        def get_predictions(model,data,device,config,outputs):
            """
            Gets the predictions of the model for the test data.

            Args:
                test_loader (DataLoader): The data loader for the test data.

            Returns:
                Tuple: A tuple containing the predicted values and the target values.
            """
            model.eval()
            preds = []
            ys = []
            model.to(device)
            data = ray.get(data)
            test_loader = DataLoader(data, batch_size=config["batch_size"], shuffle=False)
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                    for i in range(outputs):
                        y_tmp = data.y[:, i]
                        out_tmp = out[:, i]
                        present_label_indices = torch.nonzero(y_tmp != 0).view(-1)
                        if len(present_label_indices) > 0:
                            out_tmp_present = torch.index_select(out_tmp, 0, present_label_indices)
                            y_tmp_present = torch.index_select(y_tmp, 0, present_label_indices)
                            preds.extend(out_tmp_present.detach().cpu().numpy())
                            ys.extend(y_tmp_present.detach().cpu().numpy())

            return preds, ys
                

        def train_model_with_ray(config,device=None,data_train=None, data_val=None,outputs=None):
            model = AttentiveFP(in_channels=23, hidden_channels=config["hidden_channels"], out_channels=outputs,
                    edge_dim=11, num_layers=config["num_layers"], num_timesteps=config["num_timesteps"],
                    dropout=0.0)
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=10**-5)

            if train.get_checkpoint():
                with train.get_checkpoint().as_directory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                        data = pickle.load(fp)
                    start_epoch = data['epoch']
                    model.load_state_dict(data['net_state_dict'])
                    optimizer.load_state_dict(data['optimizer_state_dict'])
            else:
                start_epoch = 0
            

            for epoch in range(1000):
                train_loss = train_step(data_train, model, optimizer, device,config,outputs)
                val_loss = validate(data_val, model, device,config,outputs)
                preds, ys = get_predictions(model, data_val, device,config,outputs)
                tau, _ = kendalltau(preds, ys)
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    with open(os.path.join(checkpoint_dir, 'data.pkl'), 'wb') as fp:
                        pickle.dump({'epoch': epoch,
                                        'net_state_dict': model.state_dict(),
                                        'optimizer_state_dict': optimizer.state_dict()}, fp)
                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    train.report({"kendall_tau": tau},
                    checkpoint=checkpoint)

            

        scheduler = ASHAScheduler(
            metric="kendall_tau",
            mode="max",
            max_t=max_num_epochs,
            grace_period=1,
            reduction_factor=2)


        if gpus_per_trial > 0:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device("cpu")

        data_train_ref = ray.put(self.train_data)
        data_val_ref = ray.put(self.val_data)
        ray.init(ignore_reinit_error=True, runtime_env={"conda": ["mtl_dc"]})

        try:
            result = tune.run(
                partial(train_model_with_ray, data_train=data_train_ref, data_val=data_val_ref,device=device,outputs=self.outputs),
                resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
                config=config,
                num_samples=num_samples,
                scheduler=scheduler,
                fail_fast=False)
        except tune.error.TuneError as e:
            logging.error(f"Error in hyperparameter tuning: {e}")

        best_trial = result.get_best_trial("kendall_tau", "max", "last")

        best_trial = result.get_best_trial("kendall_tau", "max", "last")
        logging.info("Best trial config: {}".format(best_trial.config))
        logging.info("Best trial final kendall_tau: {}".format(best_trial.last_result["kendall_tau"]))

        #load model from best trial
        best_trained_model = AttentiveFP(in_channels=23, hidden_channels=best_trial.config["hidden_channels"], out_channels=self.outputs,
                    edge_dim=11, num_layers=best_trial.config["num_layers"], num_timesteps=best_trial.config["num_timesteps"],
                    dropout=0.0)
        optimizer = Adam(best_trained_model.parameters(), lr=best_trial.config["lr"], weight_decay=10 ** -5)
        val_loader = DataLoader(self.val_data, batch_size=best_trial.config["batch_size"], shuffle=False)
        with result.get_best_checkpoint(trial=best_trial, metric="kendall_tau", mode="max").as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                data = pickle.load(fp)
            start_epoch = data['epoch']
            best_trained_model.load_state_dict(data['net_state_dict'])
            optimizer.load_state_dict(data['optimizer_state_dict'])
        preds, ys = get_predictions(best_trained_model, data_val_ref, device,best_trial.config,self.outputs)
        tau, _ = kendalltau(preds, ys)
        logging.info(f'Kendall Tau: {tau}')
        self.set_model_params(best_trial.config)
        self.model = best_trained_model
        self.save_model()
        self.save_hp_configs()
    
    
        
    def eval_best_HPs(self,data_val):
            
            best_trial = self.HP_tuning_result.get_best_trial("kendall_tau", "max", "last")
            best_trained_model = AttentiveFP(in_channels=23, hidden_channels=best_trial.config["hidden_channels"], out_channels=self.outputs,
                        edge_dim=11, num_layers=best_trial.config["num_layers"], num_timesteps=best_trial.config["num_timesteps"],
                        dropout=0.0)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            optimizer = Adam(best_trained_model.parameters(), lr=best_trial.config["lr"], weight_decay=10 ** -5)
            val_loader = DataLoader(data_val, batch_size=best_trial.config["batch_size"], shuffle=False)
            with self.HP_tuning_result.get_best_checkpoint(trial=best_trial, metric="kendall_tau", mode="max").as_directory() as checkpoint_dir:
                with open(os.path.join(checkpoint_dir, 'data.pkl'), 'rb') as fp:
                    data = pickle.load(fp)
                start_epoch = data['epoch']
                best_trained_model.load_state_dict(data['net_state_dict'])
                optimizer.load_state_dict(data['optimizer_state_dict'])
            preds, ys = self.get_predictions(model = best_trained_model,test_loader = val_loader,device = device)
            tau, _ = kendalltau(preds, ys)
            logging.info(f'Kendall Tau: {tau}')
            self.set_model_params(best_trial.config)