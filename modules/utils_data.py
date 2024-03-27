import inspect
import os
import numpy as np
import torch
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from serenityff.charge.utils import Atom, Bond, Molecule
from serenityff.charge.gnn.utils import MolGraphConvFeaturizer
from torch_geometric.data import Data
from torch_geometric.typing import OptTensor
from tqdm import tqdm
from functools import reduce
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from typing import Any, Iterable, List, Tuple, Union, Optional

# Constants
DEFAULT_HYBRIDIZATION_SET = ["SP", "SP2", "SP3"]
DEFAULT_TOTAL_DEGREE_SET = [0, 1, 2, 3, 4, 5]
DEFAULT_BOND_TYPE_SET = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
DEFAULT_BOND_STEREO_SET = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
allowable_set = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H"]
default_properties = ['LogVP', 'LogP', 'LogOH', 'LogBCF', 'LogHalfLife', 'BP', 'Clint', 'FU', 'LogHL', 'LogKmHL', 'LogKOA', 'LogKOC', 'MP', 'LogMolar']


class GraphData:
    """GraphData class

    This data class is almost same as `deepchems graphdata
    <https://github.com/deepchem/deepchem/blob/master/\
        deepchem/feat/graph_data.py>`_.

    Attributes
    ----------
    node_features: np.ndarray
      Node feature matrix with shape [num_nodes, num_node_features]
    edge_index: np.ndarray, dtype int
      Graph connectivity in COO format with shape [2, num_edges]
    edge_features: np.ndarray, optional (default None)
      Edge feature matrix with shape [num_edges, num_edge_features]
    node_pos_features: np.ndarray, optional (default None)
      Node position matrix with shape [num_nodes, num_dimensions].
    num_nodes: int
      The number of nodes in the graph
    num_node_features: int
      The number of features per node in the graph
    num_edges: int
      The number of edges in the graph
    num_edges_features: int, optional (default None)
      The number of features per edge in the graph

    Examples
    --------
    >>> import numpy as np
    >>> node_features = np.random.rand(5, 10)
    >>> edge_index = np.array([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]], dtype=np.int64)
    >>> graph = GraphData(node_features=node_features, edge_index=edge_index)
    """

    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_pos_features: Optional[np.ndarray] = None,
    ):
        """
        Parameters
        ----------
        node_features: np.ndarray
          Node feature matrix with shape [num_nodes, num_node_features]
        edge_index: np.ndarray, dtype int
          Graph connectivity in COO format with shape [2, num_edges]
        edge_features: np.ndarray, optional (default None)
          Edge feature matrix with shape [num_edges, num_edge_features]
        node_pos_features: np.ndarray, optional (default None)
          Node position matrix with shape [num_nodes, num_dimensions].
        """
        # validate params
        if isinstance(node_features, np.ndarray) is False:
            raise TypeError("node_features must be np.ndarray.")

        if isinstance(edge_index, np.ndarray) is False:
            raise TypeError("edge_index must be np.ndarray.")
        elif issubclass(edge_index.dtype.type, np.integer) is False:
            raise TypeError("edge_index.dtype must contains integers.")
        elif edge_index.shape[0] != 2:
            raise ValueError("The shape of edge_index is [2, num_edges].")
        elif np.max(edge_index) >= len(node_features):
            raise ValueError("edge_index contains the invalid node number.")

        if edge_features is not None:
            if isinstance(edge_features, np.ndarray) is False:
                raise TypeError("edge_features must be np.ndarray or None.")
            elif edge_index.shape[1] != edge_features.shape[0]:
                raise ValueError(
                    "The first dimension of edge_features must be the \
                          same as the second dimension of edge_index."
                )

        if node_pos_features is not None:
            if isinstance(node_pos_features, np.ndarray) is False:
                raise TypeError("node_pos_features must be np.ndarray or None.")
            elif node_pos_features.shape[0] != node_features.shape[0]:
                raise ValueError(
                    "The length of node_pos_features must be the same as the \
                          length of node_features."
                )

        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.node_pos_features = node_pos_features
        self.num_nodes, self.num_node_features = self.node_features.shape
        self.num_edges = edge_index.shape[1]
        if self.edge_features is not None:
            self.num_edge_features = self.edge_features.shape[1]

class CustomGraphData(GraphData):
    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        edge_features: Optional[np.ndarray] = None,
        node_pos_features: Optional[np.ndarray] = None,
    ):
        super().__init__(
            node_features,
            edge_index,
            edge_features,
            node_pos_features,
        )

    def to_pyg_graph(self):
        """Convert to PyTorch Geometric graph data instance

        Returns
        -------
        torch_geometric.data.Data
        Graph data for PyTorch Geometric

        Note
        ----
        This method requires PyTorch Geometric to be installed.
        """

        edge_features = self.edge_features
        if edge_features is not None:
            edge_features = torch.from_numpy(self.edge_features).float()
        node_pos_features = self.node_pos_features
        if node_pos_features is not None:
            node_pos_features = torch.from_numpy(self.node_pos_features).float()
        return CustomData(
            x=torch.from_numpy(self.node_features).float(),
            edge_index=torch.from_numpy(self.edge_index).long(),
            edge_attr=edge_features,
            pos=node_pos_features,
        )
    

"""_
This file is necessary to eliminate the packages dependecy on deepchem.
Most of the functionality is adapted from deepchem.
https://github.com/deepchem/deepchem/tree/master/deepchem/feat
"""
class Featurizer(object):
    """Abstract class for calculating a set of features for a datapoint.

    This class is abstract and cannot be invoked directly. You'll
    likely only interact with this class if you're a developer. In
    that case, you might want to make a child class which
    implements the `_featurize` method for calculating features for
    a single datapoints if you'd like to make a featurizer for a
    new datatype.
    """

    def featurize(self, datapoints: Iterable[Any], log_every_n: int = 1000, **kwargs) -> np.ndarray:
        """Calculate features for datapoints.

        Parameters
        ----------
        datapoints: Iterable[Any]
        A sequence of objects that you'd like to featurize. Subclassses of
        `Featurizer` should instantiate the `_featurize` method that featurizes
        objects in the sequence.
        log_every_n: int, default 1000
        Logs featurization progress every `log_every_n` steps.

        Returns
        -------
        np.ndarray
        A numpy array containing a featurized representation of `datapoints`.
        """
        datapoints = list(datapoints)
        features = []
        for i, point in enumerate(datapoints):
            if i % log_every_n == 0:
                pass
            try:
                features.append(self._featurize(point, **kwargs))
            except Exception as e:
                print(e)
                features.append(np.array([]))

        return np.asarray(features)

    def __call__(self, datapoints: Iterable[Any], **kwargs):
        """Calculate features for datapoints.

        `**kwargs` will get passed directly to `Featurizer.featurize`

        Parameters
        ----------
        datapoints: Iterable[Any]
        Any blob of data you like. Subclasss should instantiate this.
        """
        return self.featurize(datapoints, **kwargs)

    def _featurize(self, datapoint: Any, **kwargs):
        """Calculate features for a single datapoint.

        Parameters
        ----------
        datapoint: Any
        Any blob of data you like. Subclass should instantiate this.
        """
        raise NotImplementedError("Featurizer is not defined.")

    def __repr__(self) -> str:
        """Convert self to repr representation.

        Returns
        -------
        str
        The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> dc.feat.CircularFingerprint(size=1024, radius=4)
        CircularFingerprint[radius=4, size=1024, chiral=False, bonds=True,
        features=False, sparse=False, smiles=False]
        >>> dc.feat.CGCNNFeaturizer()
        CGCNNFeaturizer[radius=8.0, max_neighbors=12, step=0.2]
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != "self"]
        args_info = ""
        for arg_name in args_names:
            value = self.__dict__[arg_name]
            # for str
            if isinstance(value, str):
                value = "'" + value + "'"
            # for list
            if isinstance(value, list):
                threshold = 10
                value = np.array2string(np.array(value), threshold=threshold)
            args_info += arg_name + "=" + str(value) + ", "
        return self.__class__.__name__ + "[" + args_info[:-2] + "]"

    def __str__(self) -> str:
        """Convert self to str representation.

        Returns
        -------
        str
        The string represents the class.

        Examples
        --------
        >>> import deepchem as dc
        >>> str(dc.feat.CircularFingerprint(size=1024, radius=4))
        'CircularFingerprint_radius_4_size_1024'
        >>> str(dc.feat.CGCNNFeaturizer())
        'CGCNNFeaturizer'
        """
        args_spec = inspect.getfullargspec(self.__init__)  # type: ignore
        args_names = [arg for arg in args_spec.args if arg != "self"]
        args_num = len(args_names)
        args_default_values = [None for _ in range(args_num)]
        if args_spec.defaults is not None:
            defaults = list(args_spec.defaults)
            args_default_values[-len(defaults) :] = defaults

        override_args_info = ""
        for arg_name, default in zip(args_names, args_default_values):
            if arg_name in self.__dict__:
                arg_value = self.__dict__[arg_name]
                # validation
                # skip list
                if isinstance(arg_value, list):
                    continue
                if isinstance(arg_value, str):
                    # skip path string
                    if "\\/." in arg_value or "/" in arg_value or "." in arg_value:
                        continue
                # main logic
                if default != arg_value:
                    override_args_info += "_" + arg_name + "_" + str(arg_value)
        return self.__class__.__name__ + override_args_info


class MolecularFeaturizer(Featurizer):
    """Abstract class for calculating a set of features for a
    molecule.

    The defining feature of a `MolecularFeaturizer` is that it
    uses SMILES strings and RDKit molecule objects to represent
    small molecules. All other featurizers which are subclasses of
    this class should plan to process input which comes as smiles
    strings or RDKit molecules.

    Child classes need to implement the _featurize method for
    calculating features for a single molecule.

    Note
    ----
    The subclasses of this class require RDKit to be installed.
    """

    def featurize(self, datapoints, log_every_n=1000, **kwargs) -> np.ndarray:
        """Calculate features for molecules.

        Parameters
        ----------
        datapoints: rdkit.Chem.rdchem.Mol / SMILES string / iterable
        RDKit Mol, or SMILES string or iterable sequence of RDKit mols/SMILES
        strings.
        log_every_n: int, default 1000
        Logging messages reported every `log_every_n` samples.

        Returns
        -------
        features: np.ndarray
        A numpy array containing a featurized representation of `datapoints`.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import rdmolfiles, rdmolops
            from rdkit.Chem.rdchem import Mol
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if "molecules" in kwargs:
            datapoints = kwargs.get("molecules")
            raise DeprecationWarning('Molecules is being phased out as a parameter, please pass "datapoints" instead.')

        # Special case handling of single molecule
        if isinstance(datapoints, str) or isinstance(datapoints, Mol):
            datapoints = [datapoints]
        else:
            # Convert iterables to list
            datapoints = list(datapoints)

        features: list = []
        for i, mol in enumerate(datapoints):
            if i % log_every_n == 0:
                pass
            try:
                if isinstance(mol, str):
                    # mol must be a RDKit Mol object, so parse a SMILES
                    mol = Chem.MolFromSmiles(mol)
                    # SMILES is unique, so set a canonical order of atoms
                    new_order = rdmolfiles.CanonicalRankAtoms(mol)
                    mol = rdmolops.RenumberAtoms(mol, new_order)

                features.append(self._featurize(mol, **kwargs))
            except Exception:
                if isinstance(mol, Chem.rdchem.Mol):
                    mol = Chem.MolToSmiles(mol)
                features.append(np.array([]))

        return np.asarray(features)


def one_hot_encode(
    val: Union[int, str],
    allowable_set: Union[List[str], List[int]],
    include_unknown_set: bool = False,
) -> List[float]:
    """One hot encoder for elements of a provided set.

    Examples
    --------
    >>> one_hot_encode("a", ["a", "b", "c"])
    [1.0, 0.0, 0.0]
    >>> one_hot_encode(2, [0, 1, 2])
    [0.0, 0.0, 1.0]
    >>> one_hot_encode(3, [0, 1, 2])
    [0.0, 0.0, 0.0]
    >>> one_hot_encode(3, [0, 1, 2], True)
    [0.0, 0.0, 0.0, 1.0]

    Parameters
    ----------
    val: int or str
      The value must be present in `allowable_set`.
    allowable_set: List[int] or List[str]
      List of allowable quantities.
    include_unknown_set: bool, default False
      If true, the index of all values not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
      An one-hot vector of val.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.

    Raises
    ------
    ValueError
      If include_unknown_set is False and `val` is not in `allowable_set`.
    """

    # init an one-hot vector
    if include_unknown_set is False:
        one_hot_legnth = len(allowable_set)
    else:
        one_hot_legnth = len(allowable_set) + 1
    one_hot = [0.0 for _ in range(one_hot_legnth)]

    try:
        one_hot[allowable_set.index(val)] = 1.0  # type: ignore
    except ValueError:
        if include_unknown_set:
            # If include_unknown_set is True, set the last index is 1.
            one_hot[-1] = 1.0
        else:
            pass
    return one_hot


def get_atom_hydrogen_bonding_one_hot(atom: Atom, hydrogen_bonding: List[Tuple[int, str]]) -> List[float]:
    """Get an one-hot feat about whether an atom accepts electrons or donates electrons.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    hydrogen_bonding: List[Tuple[int, str]]
      The return value of `construct_hydrogen_bonding_info`.
      The value is a list of tuple `(atom_index, hydrogen_bonding)` like (1, "Acceptor").

    Returns
    -------
    List[float]
      A one-hot vector of the ring size type. The first element
      indicates "Donor", and the second element indicates "Acceptor".
    """
    one_hot = [0.0, 0.0]
    atom_idx = atom.GetIdx()
    for hydrogen_bonding_tuple in hydrogen_bonding:
        if hydrogen_bonding_tuple[0] == atom_idx:
            if hydrogen_bonding_tuple[1] == "Donor":
                one_hot[0] = 1.0
            elif hydrogen_bonding_tuple[1] == "Acceptor":
                one_hot[1] = 1.0
    return one_hot


def get_atom_total_degree_one_hot(
    atom: Atom,
    allowable_set: List[int] = [0, 1, 2, 3, 4, 5],
    include_unknown_set: bool = True,
) -> List[float]:
    """Get an one-hot feature of the degree which an atom has.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object
    allowable_set: List[int]
      The degree to consider. The default set is `[0, 1, ..., 5]`
    include_unknown_set: bool, default True
      If true, the index of all types not in `allowable_set` is `len(allowable_set)`.

    Returns
    -------
    List[float]
      A one-hot vector of the degree which an atom has.
      If `include_unknown_set` is False, the length is `len(allowable_set)`.
      If `include_unknown_set` is True, the length is `len(allowable_set) + 1`.
    """
    return one_hot_encode(atom.GetTotalDegree(), allowable_set, include_unknown_set)


def get_atom_partial_charge(atom: Atom) -> List[float]:
    """Get a partial charge of an atom.

    Parameters
    ---------
    atom: rdkit.Chem.rdchem.Atom
      RDKit atom object

    Returns
    -------
    List[float]
      A vector of the parital charge.

    Notes
    -----
    Before using this function, you must calculate `GasteigerCharge`
    like `AllChem.ComputeGasteigerCharges(mol)`.
    """
    DASH_charge = atom.GetProp("charge")
    if DASH_charge in ["-nan", "nan", "-inf", "inf"]:
        DASH_charge = 0.0
    return [float(DASH_charge)]


def _construct_atom_feature(
    atom: Atom,
    h_bond_infos: List[Tuple[int, str]],
    use_partial_charge: bool,
    allowable_set: List[str],
) -> np.ndarray:
    """
    Constructs an atom feature from a RDKit atom object.
    In this case creates one hot features for the Attentive FP model.
    The only thing changed is, that it now passes information about
    what atom types you want to have a feature for, given in allowable_set.

    Args:
        atom (RDKitAtom): RDKit atom object
        h_bond_infos (List[Tuple[int, str]]): A list of tuple `(atom_index, hydrogen_bonding_type)`.
        Basically, it is expected that this value is the return value
        of `construct_hydrogen_bonding_info`.The `hydrogen_bonding_type` value is "Acceptor"
        or "Donor".use_partial_charge (bool): Whether to use partial charge data or not.
        allowable_set (List[str]): List of Atoms you want to have features for.

    Returns:
        np.ndarray: A one-hot vector of the atom feature.
    """
    atom_type = one_hot_encode(atom.GetSymbol(), allowable_set, True)
    formal_charge = [float(atom.GetFormalCharge())]
    hybridization = one_hot_encode(str(atom.GetHybridization()), DEFAULT_HYBRIDIZATION_SET, False)
    # remove
    # acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = [float(atom.GetIsAromatic())]
    degree = one_hot_encode(atom.GetTotalDegree(), DEFAULT_TOTAL_DEGREE_SET, True)
    atom_feat = np.concatenate(
        [
            atom_type,
            formal_charge,
            hybridization,
            #    acceptor_donor,
            aromatic,
            degree,
        ]
    )
    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, np.array(partial_charge)])
    return atom_feat


def _construct_bond_feature(bond: Bond) -> np.ndarray:
    """
    Construct a bond feature from a RDKit bond object. Not changed.

    Args:
        bond (RDKitBond): RDKit bond object

    Returns:
        np.ndarray: A one-hot vector of the bond feature.
    """
    bond_type = one_hot_encode(str(bond.GetBondType()), DEFAULT_BOND_TYPE_SET, False)
    same_ring = [int(bond.IsInRing())]
    conjugated = [int(bond.GetIsConjugated())]
    stereo = one_hot_encode(str(bond.GetStereo()), DEFAULT_BOND_STEREO_SET, True)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])


class _ChemicalFeaturesFactory:
    """This is a singleton class for RDKit base features."""

    _instance = None

    @classmethod
    def get_instance(cls):
        try:
            from rdkit import RDConfig
            from rdkit.Chem import ChemicalFeatures
        except ModuleNotFoundError:
            raise ImportError("This class requires RDKit to be installed.")

        if not cls._instance:
            fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            cls._instance = ChemicalFeatures.BuildFeatureFactory(fdefName)
        return cls._instance


def construct_hydrogen_bonding_info(mol: Molecule) -> List[Tuple[int, str]]:
    """Construct hydrogen bonding infos about a molecule.

    Parameters
    ---------
    mol: rdkit.Chem.rdchem.Mol
      RDKit mol object

    Returns
    -------
    List[Tuple[int, str]]
      A list of tuple `(atom_index, hydrogen_bonding_type)`.
      The `hydrogen_bonding_type` value is "Acceptor" or "Donor".
    """
    factory = _ChemicalFeaturesFactory.get_instance()
    feats = factory.GetFeaturesForMol(mol)
    hydrogen_bonding = []
    for f in feats:
        hydrogen_bonding.append((f.GetAtomIds()[0], f.GetFamily()))
    return hydrogen_bonding


class MolGraphConvFeaturizer_charges(MolecularFeaturizer):
    """
    Same as original by deepchem, excyept, that you now can give an allowable set,
    that determines for which atom types a feature in the one hot vector is created.


    This class is a featurizer of general graph convolution networks for molecules.
    The default node(atom) and edge(bond) representations are based on
    `WeaveNet paper <https://arxiv.org/abs/1603.00856>`_.
    If you want to use your own representations, you could use this class as a guide
    to define your original Featurizer. In many cases, it's enough
    to modify return values of `construct_atom_feature` or `construct_bond_feature`.
    The default node representation are constructed by concatenating the following values,
    and the feature length is 30.
    - Atom type: A one-hot vector of this atom, "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "other".
    - Formal charge: Integer electronic charge.
    - Hybridization: A one-hot vector of "sp", "sp2", "sp3".
    - Hydrogen bonding: A one-hot vector of whether this atom is a hydrogen bond donor or acceptor.
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Chirality: A one-hot vector of the chirality, "R" or "S". (Optional)
    - Partial charge: Calculated partial charge. (Optional)
    The default edge representation are constructed by concatenating the following values,
    and the feature length is 11.
    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration of a bond.
    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.
    Examples
    --------
    >>> smiles = ["C1CCC1", "C1=CC=CN=C1"]
    >>> featurizer = MolGraphConvFeaturizer(use_edges=True)
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.CustomGraphData'>
    >>> out[0].num_node_features
    30
    >>> out[0].num_edge_features
    11
    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
       Journal of computer-aided molecular design 30.8 (2016):595-608.
    Note
    ----
    This class requires RDKit to be installed.
    """

    def __init__(
        self,
        use_edges: bool = False,
        use_partial_charge: bool = True,
    ):
        """
        Parameters
        ----------
        use_edges: bool, default False
          Whether to use edge features or not.
        use_partial_charge: bool, default False
          Whether to use partial charge data or not.
          If True, this featurizer computes gasteiger charges.
          Therefore, there is a possibility to fail to featurize for some molecules
          and featurization becomes slow.
        """
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge

    def _featurize(self, datapoint: Molecule, allowable_set: List[str], **kwargs) -> CustomGraphData:
        """Calculate molecule graph features from RDKit mol object.
        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
          RDKit mol object.
        allowable_set: List[str]
          List of atoms you want a feature for in the atom feature vector
        Returns
        -------
        graph: CustomGraphData
          A molecule graph with some features.
        """
        assert (
            datapoint.GetNumAtoms() > 1
        ), "More than one atom should be present in the molecule for this featurizer to work."
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning('Mol is being phased out as a parameter, please pass "datapoint" instead.')

        if self.use_partial_charge:
            try:
                datapoint.GetAtomWithIdx(0).GetProp("_GasteigerCharge")
            except KeyError:
                # If partial charges were not computed
                try:
                    ComputeGasteigerCharges(datapoint)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        # h_bond_infos = construct_hydrogen_bonding_info(datapoint)
        h_bond_infos = [(i, "Donor") for i in range(datapoint.GetNumAtoms())]
        atom_features = np.asarray(
            [
                _construct_atom_feature(
                    atom,
                    h_bond_infos,
                    self.use_partial_charge,
                    allowable_set=allowable_set,
                )
                for atom in datapoint.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in datapoint.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return CustomGraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features,
        )


class CustomData(Data):
    """
    Data Class holding the pyorch geometric molecule graphs.
    Similar to pyg's data class but with two extra attributes,
    being smiles and molecule_charge.
    """

    def __init__(
        self,
        x: OptTensor = None,
        edge_index: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        pos: OptTensor = None,
        smiles: str = None,
        molecule_charge: int = None,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            pos=pos,
            smiles=smiles,
            molecule_charge=molecule_charge,
        )

    @property
    def smiles(self) -> Union[str, None]:
        return self["smiles"] if "smiles" in self._store else None

    @property
    def molecule_charge(self) -> Union[int, None]:
        return self["molecule_charge"] if "molecule_charge" in self._store else None

    def __setattr__(self, key: str, value: Any):
        if key == "smiles":
            return self._set_smiles(value)
        elif key == "molecule_charge":
            return self._set_molecule_charge(value)
        else:
            return super().__setattr__(key, value)

    def _set_smiles(self, value: str) -> None:
        """
        Workaround for the ._store that is implemented in pytorch geometrics data.

        Args:
            value (str): smiles to be set

        Raises:
            TypeError: if value not convertable to string.
        """
        if isinstance(value, str) or value is None:
            return super().__setattr__("smiles", value)
        else:
            raise TypeError("Attribute smiles has to be of type string")

    def _set_molecule_charge(self, value: int) -> None:
        """
        Workaround for the ._store that is implemented in pytorch geometrics data.

        Args:
            value (int): molecule charge to be set.

        Raises:
            TypeError: if value not integer.

        """
        if isinstance(value, int):
            return super().__setattr__("molecule_charge", torch.tensor([value], dtype=int))
        elif value is None:
            return super().__setattr__("molecule_charge", None)
        elif isinstance(value, float) and value.is_integer():
            return super().__setattr__("molecule_charge", torch.tensor([int(value)], dtype=int))
        else:
            raise TypeError("Value for charge has to be an int!")


def get_graph_from_mol(
    mol: Molecule,
    index: int,
    allowable_set: Optional[List[str]] = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "H",
    ],
    charges: Optional[bool] = False,
    no_y: Optional[bool] = False,
) -> CustomData:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.
    The graph contains following features:
        > Node Features:
            > Atom Type (as specified in allowable set)
            > formal_charge
            > hybridization
            > H acceptor_donor
            > aromaticity
            > degree
        > Edge Features:
            > Bond type
            > is in ring
            > is conjugated
            > stereo
    Args:
        mol (Molecule): rdkit molecule
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """
    if charges:
        grapher = MolGraphConvFeaturizer_charges(use_edges=True)
    else:
        grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if not no_y:
        graph.y = torch.tensor(
            [float(x) for x in mol.GetProp("MBIScharge").split("|")],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    #TODO: Check if batch is needed, otherwise this could lead to a problem if all batches are set to 0
    # Batch will be overwritten by the DataLoader class
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph

def get_graph_from_mol(
    mol: Molecule,
    index: int,
    allowable_set: Optional[List[str]] = [
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
        "H",
    ],
    no_y: Optional[bool] = False,
) -> CustomData:
    """
    Creates an pytorch_geometric Graph from an rdkit molecule.
    The graph contains following features:
        > Node Features:
            > Atom Type (as specified in allowable set)
            > formal_charge
            > hybridization
            > H acceptor_donor
            > aromaticity
            > degree
        > Edge Features:
            > Bond type
            > is in ring
            > is conjugated
            > stereo
    Args:
        mol (Molecule): rdkit molecule
        allowable_set (Optional[List[str]], optional): List of atoms to be \
            included in the feature vector. Defaults to \
                [ "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "H", ].

    Returns:
        CustomData: pytorch geometric Data with .smiles as an extra attribute.
    """
    grapher = MolGraphConvFeaturizer(use_edges=True)
    graph = grapher._featurize(mol, allowable_set).to_pyg_graph()
    if not no_y:
        graph.y = torch.tensor(
            [float(x) for x in mol.GetProp("MBIScharge").split("|")],
            dtype=torch.float,
        )
    else:
        graph.y = torch.tensor(
            [0 for _ in mol.GetAtoms()],
            dtype=torch.float,
        )
    #TODO: Check if batch is needed, otherwise this could lead to a problem if all batches are set to 0
    # Batch will be overwritten by the DataLoader class
    graph.batch = torch.tensor([0 for _ in mol.GetAtoms()], dtype=int)
    graph.molecule_charge = Chem.GetFormalCharge(mol)
    graph.smiles = Chem.MolToSmiles(mol, canonical=True)
    graph.sdf_idx = index
    return graph


def read_sdf(file_path,prop_name,conditions = None):
    """
    Reads a Structure-Data File (SDF) and extracts the SMILES representation and a specified property for each molecule.

    Parameters:
    file_path (str): The path to the SDF file.
    prop_name (str): The name of the property to extract from each molecule.
    conditions (dict, optional): A dictionary of conditions to apply when reading the file. Not implemented in this function.

    Returns:
    list: A list of dictionaries, where each dictionary contains the SMILES representation and the specified property of a molecule.
    If an error occurs while reading the file, the function returns None.

    """
    print(f'Reading {file_path}')
    try:
        suppl = Chem.SDMolSupplier(file_path)
        data = []
        for mol in tqdm(suppl):
            if mol is not None:
                smiles = Chem.MolToSmiles(mol)
                prop_val = float(mol.GetProp(f'{prop_name}')) if mol.HasProp(f'{prop_name}') else None
                data.append({'SMILES': smiles, prop_name: prop_val})
        return data
    except OSError:
        print(f'Error reading {file_path}')
        return None
    

def load_data(overwrite=False,prop_list = default_properties):
    """
    Loads data from a CSV file or reads from SDF files if the CSV file is not found or if overwrite is True.

    Parameters:
    overwrite (bool): If True, the function will ignore the CSV file and read from the SDF files. Default is False.
    prop_list (list): A list of properties to extract from the SDF files. Default is the list of default properties.

    Returns:
    DataFrame: A pandas DataFrame containing the SMILES representation and the specified properties of the molecules.

    The DataFrame is loaded from a CSV file if it exists and overwrite is False. If the CSV file does not exist or if overwrite is True, the function reads the SDF files, extracts the SMILES representation and the specified properties for each molecule, removes duplicates based on the SMILES representation, and merges the data into a single DataFrame.
    """
    if not overwrite:
        try:
            # Load data from file
            print('Loading data from file')
            df = pd.read_csv('OPERA_Data/OPERA_properties.csv')
            return df
        except FileNotFoundError:
            print('File not found, reading from SDF files')

    file_paths = {
        'LogVP': 'OPERA_Data/VP_QR.sdf',
        'LogP': 'OPERA_Data/LogP_QR.sdf',
        'LogOH': 'OPERA_Data/AOH_QR.sdf',
        'LogBCF': 'OPERA_Data/BCF_QR.sdf',
        'LogHalfLife': 'OPERA_Data/Biodeg_QR.sdf',
        'BP': 'OPERA_Data/BP_QR.sdf',
        'Clint': 'OPERA_Data/Clint_QR.sdf',
        'FU': 'OPERA_Data/FU_QR.sdf',
        'LogHL': 'OPERA_Data/HL_QR.sdf',
        'LogKmHL': 'OPERA_Data/KM_QR.sdf',
        'LogKOA': 'OPERA_Data/KOA_QR.sdf',
        'LogKOC': 'OPERA_Data/KOC_QR.sdf',
        'MP': 'OPERA_Data/MP_QR.sdf',
        'LogMolar': 'OPERA_Data/WS_QR.sdf',
    }

    # Read SDF files and extract data
    data_dict = {prop: read_sdf(path, prop) for prop, path in file_paths.items()}

    # Create Pandas dataframes
    df_dict = {prop: pd.DataFrame(data) for prop, data in data_dict.items()}

    #for eaach df in df_dict, remove duplicates
    for df in df_dict.values():
        df.drop_duplicates(subset='SMILES', inplace=True)

    df_combined = reduce(lambda left, right: pd.merge(left, right, on='SMILES', how='outer'), [df[['SMILES', prop]] for prop, df in df_dict.items()])
    df_combined = df_combined[['SMILES'] + list(df_dict.keys())]
    return df_combined


def plot_property_histograms(df):
    """
    Plots histograms of the properties in the given DataFrame.

    Parameters:
    df (DataFrame): A pandas DataFrame where each column represents a property. The first column is ignored.

    The function creates a 3x5 grid of subplots and plots a histogram for each property in the DataFrame. The title of each subplot is the name of the property.
    """
    fig, axes = plt.subplots(2, 7, figsize=(20, 5.5))
    for i, (col, ax) in enumerate(zip(df.columns[1:], axes.flatten())):
        sns.histplot(df[col], ax=ax, kde=True, bins=50, color='skyblue', edgecolor='black')
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('')
    plt.tight_layout()
    plt.show()

def scale_props(df,prop_list = default_properties):
    """
    Scales the properties in the given DataFrame using MinMaxScaler.

    Parameters:
    df (DataFrame): A pandas DataFrame where each column represents a property.
    prop_list (list): A list of properties to scale. Default is the list of default properties.

    Returns:
    DataFrame, MinMaxScaler: A new DataFrame where the specified properties have been scaled to the range [0, 1], and the fitted MinMaxScaler object.

    The function creates a copy of the input DataFrame, scales the specified properties using MinMaxScaler, and replaces the original properties with the scaled properties in the copied DataFrame.
    """
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[prop_list] = scaler.fit_transform(df_scaled[prop_list])
    return df_scaled,scaler

def compare_distributions(df_scaled, train, test, figsize=(20,5.5)):
    """
    Compares the distributions of properties in the train and test datasets.

    Parameters:
    df_scaled (DataFrame): A pandas DataFrame where each column represents a property. The first column is ignored.
    train (DataFrame): The training dataset.
    test (DataFrame): The testing dataset.
    figsize (tuple): The size of the figure to create. Default is (15, 15).

    The function checks if the distribution of each property is similar in the train and test datasets using the Mann-Whitney U test. It then creates a grid of subplots and plots a boxplot for each property in the train and test datasets. The title of each subplot is the name of the property. The y label is 'Normalized value'.
    """
    # Check if for each property the distribution is similar in train and test
    for col in df_scaled.columns[1:]:
        train_tmp = train[col].dropna()
        test_tmp = test[col].dropna()
        print(f'{col}: {mannwhitneyu(train_tmp,test_tmp)}')

    num_cols = len(df_scaled.columns[1:])
    num_rows = num_cols // 7 if num_cols % 7 == 0 else num_cols // 7 + 1
    fig, axes = plt.subplots(num_rows, 7, figsize=figsize)

    for i, col in enumerate(df_scaled.columns[1:]):
        train_tmp = train[col].dropna()
        test_tmp = test[col].dropna()
        row = i // 7
        col_i = i % 7
        ax = axes[row, col_i]
        ax.boxplot(x=[train_tmp,test_tmp],labels=['train','test'])
        ax.set_xticklabels(['train', 'test'])
        ax.set_ylabel('Normalized value')
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

def get_graphs(df,dash_charges=False,scaled =True,test=False, save_graphs = False):
    """
    Get molecular graphs from the given DataFrame.
    
    Parameters:
    df (DataFrame): A pandas DataFrame containing the SMILES representation and the specified properties of the molecules.
    dash_charges (bool, optional): Whether to use dash charges. Defaults to False.
    scaled (bool, optional): Whether to scale the graphs. Defaults to True.
    test (bool, optional): Whether to get the train or test graphs. Defaults to False (train graphs)
    save_graphs (bool, optional): Whether to save the graphs to a file. Defaults to False.
    
    Returns:
    mol_graphs: The molecular graphs.
    """
    graphprops = []
    if dash_charges:
        graphprops.append('dash_charges')
    if scaled:
        graphprops.append('scaled')
    else:
        graphprops.append('unscaled')
    if test:
        graphprops.append('test')
    else:
        graphprops.append('train')

    try:
        mol_graphs = torch.load(f'mol_graphs_{"_".join(graphprops)}.pt')
        print('Loading previously created graphs')
        return mol_graphs
    except FileNotFoundError:
        print('Creating new graphs')

    all_mols = [Chem.MolFromSmiles(smi) for smi in df['SMILES']]
    mols = [m for m in all_mols if m.GetNumAtoms() > 1]
    error_mol = [m for m in all_mols if m.GetNumAtoms() <= 1]
    indices_to_drop_size = [all_mols.index(m) for m in error_mol]
     
    if dash_charges:
        from serenityff.charge.tree.dash_tree import DASHTree
        tree = DASHTree(tree_folder_path='props2') #add path to tree_folder_path if needed ex. tree_folder_path='/localhome/..../props2'
        tmp_mols = mols 
        mols = []
        error_mols_charges = []
        for m in tqdm(tmp_mols):
            try:
                mol = Chem.AddHs(m, addCoords=True)
                charges = tree.get_molecules_partial_charges(mol,chg_std_key='std',chg_key='result')["charges"]
            except:
                error_mols_charges.append(m)
                continue
            for i,atom in enumerate(mol.GetAtoms()):
                atom.SetDoubleProp('charge',charges[i])
            mols.append(mol)
        indices_to_drop_charges = [all_mols.index(m) for m in error_mols_charges]
        indices_to_drop_total = list(set(indices_to_drop_size + indices_to_drop_charges))
        print(len(indices_to_drop_total), len(indices_to_drop_size), len(indices_to_drop_charges))
        #get the smiles of indices_to_drop_total
        smiles_to_drop = [df['SMILES'].iloc[i] for i in indices_to_drop_total]
        if indices_to_drop_total:
            #drop these smiles from the dataframe
            df = df[~df['SMILES'].isin(smiles_to_drop)]
    
    else:
        if indices_to_drop_size:
            print(f'Caution! {len(indices_to_drop_size)} Mols dropped')
            df = df.drop(indices_to_drop_size)

    ys = df.iloc[:, 1:].values
    ys = np.nan_to_num(ys, nan=-1)
    y = torch.tensor(ys, dtype=torch.float32)
    y = y.unsqueeze(1)
    assert len(mols) == len(y)
    mol_graphs = [get_graph_from_mol(mol,i, allowable_set,no_y=True) for i,mol in enumerate(mols)]
    SMILES = df['SMILES'].tolist()
    assert len(mol_graphs) == len(y) == len(SMILES)
    for i in range(len(mol_graphs)):
        mol_graphs[i].SMILES = SMILES[i]
        mol_graphs[i].y = y[i]
    if save_graphs:
        torch.save(mol_graphs, f'mol_graphs_{"_".join(graphprops)}.pt')
    return mol_graphs