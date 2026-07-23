# SPDX-FileCopyrightText: 2025 Stanford University and the project authors (see CONTRIBUTORS.md)
#
# SPDX-License-Identifier: Apache-2.0


"""
This file should contian an edge protocol. 

"""

from typing import Union,Protocol,List
from ..node.node_protocol import NodeProtocol, BaseData,RecursiveDict
from dataclasses import dataclass
from uuid import UUID,uuid4




# TODO: deprecate this, 
@dataclass
class EdgeData(BaseData):
    """ Class for indicting data stored in node. """
    id: Union[int, UUID]
    data: RecursiveDict



class EdgeProtocol(Protocol):
    "Edge Protocol"
    def __init__(self,edge_id:Union[int, UUID], source:Union[NodeProtocol,List[NodeProtocol]], destination:Union[NodeProtocol]):
        self.id=edge_id
        self.source = source
        self.target = destination
    
    def __repr__(self) -> str:
        return f"Edge({self.source}, {self.target})"

class DynamicDataEdge(EdgeProtocol):
    """Dynamic Data Edge, a read only edge, with the ability 

    Args:
        EdgeProtocol (_type_): _description_
    """
    def __init__(self, source:Union[NodeProtocol], destination:Union[NodeProtocol],data:Union[EdgeData,dict],**kwargs):
        """
        Summary 

        Args:
            source (Union[NodeProtocol]): 
            destination (Union[NodeProtocol]): _description_
            data (Union[EdgeData,dict]): _description_
        """
        self.id = kwargs.pop("edge_id", uuid4())  # required, but fallback-safe
        self.source = source
        self.target = destination
        self.data = data
    
    def get_data(self) -> Union[EdgeData, dict, None]:
        "getter function for data"
        return self.data
    

class OrderedEdge(EdgeProtocol):
    "Ordered Edge"
    def __init__(self, source:Union[NodeProtocol],
                 destination:Union[NodeProtocol],
                 order_index:Union[int]):
        super().__init__(source, destination)
        self.order_index = order_index
    def get_data(self) -> Union[dict, None]:
        pass
    def __repr__(self) -> str:
        return f"OrderedEdge({self.source}, {self.target}, {self.order_index})"


class ValueChangeEdge(EdgeProtocol):
    """Edge carrying an objective value-change plus the collapsible-branch metadata.

    An edge encodes what changed between two patient states: `change` is a
    list of dicts (variable, variable_cui, direction, from_value, to_value, delta,
    unit). Branch fields (branch_flag / branch_role / branch_id / branch_depth)
    place the edge in the collapsible bracket structure. Kept intentionally close
    to the pipeline's JSON edge dict so conversion is lossless.
    """

    def __init__(self, source, destination, change=None, content="",
                 branch_flag=False, branch_role="none", branch_id=None,
                 branch_depth=0, **kwargs):
        self.id = kwargs.pop("edge_id", uuid4())
        self.source = source
        self.target = destination
        self.content = content
        self.change = change or []
        self.branch_flag = branch_flag
        self.branch_role = branch_role
        self.branch_id = branch_id
        self.branch_depth = branch_depth

    def get_data(self) -> dict:
        "Return the edge as a JSON-serialisable dict."
        return {
            "edge_id": self.id,
            "content": self.content,
            "change": self.change,
            "branch_flag": self.branch_flag,
            "branch_role": self.branch_role,
            "branch_id": self.branch_id,
            "branch_depth": self.branch_depth,
        }

    def __repr__(self) -> str:
        return f"ValueChangeEdge({self.source}->{self.target}, role={self.branch_role})"

