"""
带证据推理功能的加权有向无环图（Evidence Reasoning DAG）实现

该模块实现了支持证据推理的有向无环图，包括：
1. 三种节点类型：AND、OR、FUSION
2. 证据推理功能：辨识框架、基本概率分配（BPA）
3. 节点可靠性计算
4. 证据传播和融合
5. 拓扑排序和路径计算

作者：Alexander with AI Assistant
日期：2025年10月5日
版本：2.0 - 对1.5版本进行了优化
"""

from collections import deque, defaultdict
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
# import math


class NodeType(Enum):
    """节点类型枚举类"""
    AND = 1      # AND类型节点：所有输入必须满足
    OR = 2       # OR类型节点：任一输入满足即可
    FUSION = 3   # FUSION类型节点：使用Dempster规则融合输入证据


@dataclass
class InputBPA:
    """FUSION节点输入BPA数据结构"""
    bpa: Dict[frozenset, float]
    reliability: float


@dataclass
class ReliabilityResult:
    """可靠性计算结果"""
    certification: float
    reliability: float


class EvidenceNode:
    """
    证据节点类
    
    属性:
        name (str): 节点名称
        frame (set): 辨识框架，包含所有可能的假设
        bpa (dict): 基本概率分配，键为假设子集，值为概率质量
        input_bpas (dict): 对于FUSION节点，存储每个输入节点的BPA及其可靠性
        reliability (float): 节点可靠性，取值范围[0,1]
        node_type (NodeType): 节点类型（AND/OR/FUSION）
        dimension (int): 节点客体的辨识框架中元素的个数
        其中，∅代表空集，Ω代表全集
    """
    
    # 常量定义
    EMPTY_SET = frozenset({'∅'})
    FULL_SET = frozenset({'Ω'})
    FLOAT_TOLERANCE = 1e-10
    
    def __init__(self, name: str, frame: Optional[Set] = None, bpa: Optional[Dict] = None, 
                 dimension: int = 20, reliability: float = 1.0, node_type: NodeType = NodeType.AND):
        """
        初始化证据节点
        """
        self.name = name
        self.frame = set(frame) if frame else set()
        self.input_bpas: Dict[str, InputBPA] = {}
        self.dimension = dimension
        self.node_type = node_type
        
        # 使用属性设置方法进行验证
        self.set_reliability(reliability)
        self.set_bpa(bpa if bpa is not None else {})
    
    def set_frame(self, frame: Set) -> None:
        """设置辨识框架"""
        self.frame = set(frame)
    
    def set_bpa(self, bpa: Dict[frozenset, float]) -> None:
        """
        设置基本概率分配
        
        异常:
            ValueError: 当BPA值之和大于1或子集不在辨识框架内时抛出
        """
        self._validate_bpa(bpa)
        self.bpa = bpa.copy()  # 创建副本避免外部修改
    
    def _validate_bpa(self, bpa: Dict[frozenset, float]) -> None:
        """验证BPA的有效性"""
        if not bpa:
            return
            
        total = sum(bpa.values())
        
        if total > 1.0 + self.FLOAT_TOLERANCE:
            raise ValueError(f"BPA值之和不能大于1，当前总和为 {total:.6f}")
        
        # 检查负值
        for subset, mass in bpa.items():
            if mass < 0:
                raise ValueError(f"BPA值不能小于0，子集 {subset} 的值为 {mass}")
        
        # 验证所有子集是否都在辨识框架内
        for subset in bpa:
            if not self._is_valid_subset(subset):
                raise ValueError(f"子集 {subset} 不在辨识框架内")
    
    def _is_valid_subset(self, subset: frozenset) -> bool:
        """检查子集是否在辨识框架内"""
        if subset in (self.EMPTY_SET, self.FULL_SET):
            return True
        return subset.issubset(self.frame)
    
    def set_input_bpa(self, input_node: str, bpa: Dict[frozenset, float], reliability: float = 1.0) -> None:
        """
        为FUSION节点设置输入节点的BPA及其可靠性
        
        异常:
            ValueError: 当节点不是FUSION类型时抛出
        """
        if self.node_type != NodeType.FUSION:
            raise ValueError("只有FUSION类型的节点可以设置输入BPA")
        
        self._validate_bpa(bpa)
        self._validate_reliability(reliability)
        
        self.input_bpas[input_node] = InputBPA(bpa=bpa.copy(), reliability=reliability)
    
    def set_reliability(self, reliability: float) -> None:
        """设置节点可靠性"""
        self._validate_reliability(reliability)
        self.reliability = reliability
    
    def _validate_reliability(self, reliability: float) -> None:
        """验证可靠性值是否有效"""
        if not 0 <= reliability <= 1:
            raise ValueError("可靠性必须在0到1之间")
    
    def set_node_type(self, node_type: NodeType) -> None:
        """设置节点类型"""
        self.node_type = node_type
    
    def set_dimension(self, dimension: int) -> None:
        """设置节点维数"""
        if dimension <= 0:
            raise ValueError("维度必须大于0")
        self.dimension = dimension
    
    def get_belief(self, hypothesis: Optional[Set] = None) -> float:
        """
        计算假设的置信函数值
        """
        hypothesis = self._get_hypothesis(hypothesis)
        discounted_bpa = self.correct_bpa(self.bpa, self.reliability)
        
        belief = 0.0
        for subset, mass in discounted_bpa.items():
            if self._is_subset_in_hypothesis(subset, hypothesis):
                belief += mass
        return belief
    
    def get_plausibility(self, hypothesis: Optional[Set] = None) -> float:
        """
        计算假设的似然函数值
        """
        hypothesis = self._get_hypothesis(hypothesis)
        discounted_bpa = self.correct_bpa(self.bpa, self.reliability)
        
        plausibility = 0.0
        for subset, mass in discounted_bpa.items():
            if self._has_intersection_with_hypothesis(subset, hypothesis):
                plausibility += mass
        return plausibility
    
    def get_spdf(self, hypothesis: Optional[Set] = None) -> float:
        """
        计算假设的类概率函数值
        """
        hypothesis = self._get_hypothesis(hypothesis)
        
        # 计算假设中含有辨识框架元素的个数
        count = sum(1 for element in self.frame 
                   if {element}.issubset(hypothesis) or hypothesis == self.FULL_SET)
        
        belief = self.get_belief(hypothesis)
        plausibility = self.get_plausibility(hypothesis)
        
        return belief + (plausibility - belief) * count / self.dimension
    
    def _get_hypothesis(self, hypothesis: Optional[Set]) -> Set:
        """获取或验证假设"""
        if hypothesis is None:
            return self.frame
        
        if not self._is_valid_hypothesis(hypothesis):
            raise ValueError("假设必须在辨识框架内")
        
        return hypothesis
    
    def _is_valid_hypothesis(self, hypothesis: Set) -> bool:
        """验证假设是否有效"""
        if hypothesis in (self.EMPTY_SET, self.FULL_SET):
            return True
        return hypothesis.issubset(self.frame)
    
    def _is_subset_in_hypothesis(self, subset: frozenset, hypothesis: Set) -> bool:
        """检查子集是否在假设中"""
        if subset == self.EMPTY_SET:
            return False
        if hypothesis == self.FULL_SET:
            return subset != self.EMPTY_SET
        return subset.issubset(hypothesis)
    
    def _has_intersection_with_hypothesis(self, subset: frozenset, hypothesis: Set) -> bool:
        """检查子集是否与假设有交集"""
        if subset == self.FULL_SET:
            return hypothesis != self.EMPTY_SET
        if subset == self.EMPTY_SET:
            return False
        if hypothesis == self.FULL_SET:
            return subset != self.EMPTY_SET
        return bool(subset.intersection(hypothesis))
    
    @classmethod
    def correct_bpa(cls, bpa: Dict[frozenset, float], reliability: float) -> Dict[frozenset, float]:
        """
        根据可靠性对BPA进行折扣
        """
        if not 0 <= reliability <= 1:
            raise ValueError("可靠性必须在0到1之间")
        
        discounted_bpa = {}
        mass_sum = 0.0
        
        for subset, mass in bpa.items():
            discounted_mass = mass * reliability
            discounted_bpa[subset] = discounted_mass
            mass_sum += discounted_mass
        
        # 将剩余的概率质量分配给全集
        discounted_bpa[cls.FULL_SET] = discounted_bpa.get(cls.FULL_SET, 0) + (1 - mass_sum)
        
        return discounted_bpa
    
    @classmethod
    def merge_evidence_dempster(cls, bpa1: Dict[frozenset, float], bpa2: Dict[frozenset, float], 
                               frame: Set) -> Dict[frozenset, float]:
        """
        使用Dempster规则合成两个BPA
        """
        # 验证BPA的有效性
        for bpa in [bpa1, bpa2]:
            for subset in bpa:
                if not cls._is_subset_valid_for_frame(subset, frame):
                    raise ValueError(f"子集 {subset} 不在辨识框架内")
        
        # 计算冲突系数K
        K = cls._calculate_conflict(bpa1, bpa2)
        
        if abs(K - 1) < cls.FLOAT_TOLERANCE:
            raise ValueError("证据完全冲突，无法合成")
        
        # 计算合成后的BPA
        combined_bpa = cls._combine_bpas(bpa1, bpa2, K)
        
        # 归一化处理
        return cls._normalize_bpa(combined_bpa)
    
    @staticmethod
    def _is_subset_valid_for_frame(subset: frozenset, frame: Set) -> bool:
        """检查子集对于辨识框架是否有效"""
        if subset in (EvidenceNode.EMPTY_SET, EvidenceNode.FULL_SET):
            return True
        return subset.issubset(frame)
    
    @staticmethod
    def _calculate_conflict(bpa1: Dict[frozenset, float], bpa2: Dict[frozenset, float]) -> float:
        """计算冲突系数K"""
        K = 0.0
        for A, mass1 in bpa1.items():
            for B, mass2 in bpa2.items():
                if EvidenceNode._are_subsets_conflicting(A, B):
                    K += mass1 * mass2
        return K
    
    @staticmethod
    def _are_subsets_conflicting(A: frozenset, B: frozenset) -> bool:
        """判断两个子集是否冲突"""
        if A == EvidenceNode.EMPTY_SET or B == EvidenceNode.EMPTY_SET:
            return not A.intersection(B)
        else:
            return (not A.intersection(B) and 
                   A != EvidenceNode.FULL_SET and 
                   B != EvidenceNode.FULL_SET)
    
    @staticmethod
    def _combine_bpas(bpa1: Dict[frozenset, float], bpa2: Dict[frozenset, float], K: float) -> Dict[frozenset, float]:
        """组合两个BPA"""
        combined_bpa = {}
        
        for A, mass1 in bpa1.items():
            for B, mass2 in bpa2.items():
                intersection = EvidenceNode._get_intersection(A, B)
                if intersection:
                    combined_bpa[intersection] = combined_bpa.get(intersection, 0) + mass1 * mass2 / (1 - K)
        
        return combined_bpa
    
    @staticmethod
    def _get_intersection(A: frozenset, B: frozenset) -> Optional[frozenset]:
        """获取两个子集的交集（处理特殊集合）"""
        if A == EvidenceNode.FULL_SET and B != EvidenceNode.EMPTY_SET:
            return B
        if B == EvidenceNode.FULL_SET and A != EvidenceNode.EMPTY_SET:
            return A
        intersection = A.intersection(B)
        return intersection if intersection else None
    
    @staticmethod
    def _normalize_bpa(bpa: Dict[frozenset, float]) -> Dict[frozenset, float]:
        """归一化BPA"""
        total = sum(bpa.values())
        
        if abs(total - 1) < EvidenceNode.FLOAT_TOLERANCE:
            return bpa  # 已经归一化
        
        if total > 1:
            # 按比例缩放
            return {subset: mass / total for subset, mass in bpa.items()}
        else:
            # 将剩余质量分配给全集
            bpa = bpa.copy()  # 避免修改原始字典
            bpa[EvidenceNode.FULL_SET] = bpa.get(EvidenceNode.FULL_SET, 0) + (1 - total)
            return bpa
    
    def has_bpa(self) -> bool:
        """检查节点是否有BPA"""
        return bool(self.bpa)
    
    def get_parent_nodes(self, graph: Dict[str, List[str]]) -> List[str]:
        """获取所有父节点"""
        return [u for u in graph if self.name in graph[u]]
    
    def __str__(self) -> str:
        """返回节点的字符串表示"""
        base_info = f"节点 {self.name}: 类型={self.node_type.name}, 可靠性={self.reliability:.2f}, 辨识框架={self.frame}"
        
        if self.node_type == NodeType.FUSION and self.input_bpas:
            input_info = [f"{node}(可靠性={data.reliability:.2f})" 
                         for node, data in self.input_bpas.items()]
            return f"{base_info}, 输入BPA数量={len(self.input_bpas)} [{', '.join(input_info)}]"
        else:
            return f"{base_info}, BPA={self.bpa}"


class EvidenceDAG:
    """
    带证据推理功能的有向无环图类
    """
    
    def __init__(self, edges: Optional[List[Tuple[str, str, float]]] = None):
        """
        初始化带证据推理功能的有向无环图
        """
        self.graph: Dict[str, List[str]] = defaultdict(list)
        self.in_degree: Dict[str, int] = defaultdict(int)
        self.edge_weights: Dict[Tuple[str, str], float] = {}
        self.nodes: Dict[str, EvidenceNode] = {}
        
        # 缓存用于性能优化
        self._topo_order_cache: Optional[List[str]] = None
        self._reliability_cache: Dict[str, ReliabilityResult] = {}
        
        if edges is not None:
            for u, v, weight in edges:
                self.add_edge(u, v, weight)
    
    def add_node(self, node_name: str, frame: Optional[Set] = None, bpa: Optional[Dict] = None,
                dimension: int = 20, reliability: float = 1.0, node_type: NodeType = NodeType.AND) -> bool:
        """
        添加证据节点
        """
        if node_name not in self.nodes:
            self.nodes[node_name] = EvidenceNode(node_name, frame, bpa, dimension, reliability, node_type)
            self.graph[node_name] = []
            self.in_degree[node_name] = 0
            self._clear_cache()
            return True
        return False
    
    def add_edge(self, u: str, v: str, weight: float = 1) -> bool:
        """
        添加从u到v的有向边，带有权重
        """
        # 确保节点存在
        self.add_node(u)
        self.add_node(v)
        
        # 检查添加这条边是否会创建环
        if self.would_create_cycle(u, v):
            return False
        
        # 添加边
        self.graph[u].append(v)
        self.edge_weights[(u, v)] = weight
        self.in_degree[v] += 1
        self._clear_cache()
        return True
    
    def _clear_cache(self) -> None:
        """清除缓存"""
        self._topo_order_cache = None
        self._reliability_cache.clear()
    
    def update_edge_weight(self, u: str, v: str, weight: float) -> bool:
        """更新边的权重"""
        if (u, v) in self.edge_weights:
            self.edge_weights[(u, v)] = weight
            self._clear_cache()
            return True
        return False
    
    def get_edge_weight(self, u: str, v: str) -> Optional[float]:
        """获取边的权重"""
        return self.edge_weights.get((u, v))
    
    def would_create_cycle(self, u: str, v: str) -> bool:
        """
        检查添加从u到v的边是否会创建环
        """
        # 如果v已经是u的祖先，则添加边(u, v)会创建环
        visited = set()
        queue = deque([v])
        
        while queue:
            node = queue.popleft()
            if node == u:
                return True
            visited.add(node)
            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return False
    
    def remove_edge(self, u: str, v: str) -> bool:
        """移除从u到v的边"""
        if u in self.graph and v in self.graph[u]:
            self.graph[u].remove(v)
            self.in_degree[v] -= 1
            if (u, v) in self.edge_weights:
                del self.edge_weights[(u, v)]
            self._clear_cache()
            return True
        return False
    
    def get_nodes(self) -> List[str]:
        """返回图中所有节点"""
        return list(self.graph.keys())
    
    def get_edges(self) -> List[Tuple[str, str, float]]:
        """返回图中所有边及其权重"""
        return [(u, v, self.edge_weights.get((u, v), 1)) 
                for u in self.graph for v in self.graph[u]]
    
    def topological_sort(self) -> List[str]:
        """
        使用Kahn算法进行拓扑排序
        """
        # 使用缓存
        if self._topo_order_cache is not None:
            return self._topo_order_cache.copy()
        
        in_degree = self.in_degree.copy()
        queue = deque([node for node in self.graph if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            
            for v in self.graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        if len(topo_order) != len(self.graph):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        self._topo_order_cache = topo_order
        return topo_order.copy()
    
    def _calculate_paths(self, start: str, end: str, find_longest: bool) -> Tuple[Optional[List[str]], float]:
        """计算最长或最短路径的通用方法"""
        if start not in self.graph or end not in self.graph:
            return None, float('-inf') if find_longest else float('inf')
            
        topo_order = self.topological_sort()
        
        # 初始化距离
        init_val = float('-inf') if find_longest else float('inf')
        dist = {node: init_val for node in self.graph}
        dist[start] = 0
        
        prev = {}
        
        # 按照拓扑顺序处理每个节点
        for node in topo_order:
            if dist[node] != init_val:
                for neighbor in self.graph[node]:
                    weight = self.edge_weights.get((node, neighbor), 1)
                    new_dist = dist[node] + weight
                    
                    if (find_longest and new_dist > dist[neighbor]) or \
                       (not find_longest and new_dist < dist[neighbor]):
                        dist[neighbor] = new_dist
                        prev[neighbor] = node
        
        # 重建路径
        if dist[end] == init_val:
            return None, init_val
        
        path = []
        current = end
        while current != start:
            path.append(current)
            current = prev.get(current)
            if current is None:
                return None, init_val
        path.append(start)
        path.reverse()
        
        return path, dist[end]
    
    def longest_path(self, start: str, end: str) -> Tuple[Optional[List[str]], float]:
        """计算从start到end的最长路径（关键路径）及其权重"""
        return self._calculate_paths(start, end, find_longest=True)
    
    def shortest_path(self, start: str, end: str) -> Tuple[Optional[List[str]], float]:
        """计算从start到end的最短路径及其权重"""
        return self._calculate_paths(start, end, find_longest=False)
    
    def combine_evidence_dempster(self, node1: str, node2: str) -> Dict[frozenset, float]:
        """
        使用Dempster规则合成两个节点的证据
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("节点不存在")
            
        ev1, ev2 = self.nodes[node1], self.nodes[node2]
        if ev1.frame != ev2.frame:
            raise ValueError("两个节点的辨识框架不一致")
            
        return EvidenceNode.merge_evidence_dempster(ev1.bpa, ev2.bpa, ev1.frame)
    
    def discount_bpa(self, bpa: Dict[frozenset, float], reliability: float) -> Dict[frozenset, float]:
        """
        根据可靠性对BPA进行折扣
        """
        return EvidenceNode.correct_bpa(bpa, reliability)
    
    def get_discounted_belief(self, node_name: str, hypothesis: Optional[Set] = None, 
                             reliability: Optional[float] = None) -> float:
        """
        计算节点经过可靠性折扣后的BPA对假设的置信度
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]
        if reliability is None:
            reliability = node.reliability
        
        # 临时修改可靠性计算，然后恢复
        original_reliability = node.reliability
        try:
            node.set_reliability(reliability)
            return node.get_belief(hypothesis)
        finally:
            node.set_reliability(original_reliability)
    
    def get_discounted_plausibility(self, node_name: str, hypothesis: Optional[Set] = None,
                                   reliability: Optional[float] = None) -> float:
        """
        计算节点经过可靠性折扣后的BPA对假设的似然度
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]
        if reliability is None:
            reliability = node.reliability
        
        # 临时修改可靠性计算，然后恢复
        original_reliability = node.reliability
        try:
            node.set_reliability(reliability)
            return node.get_plausibility(hypothesis)
        finally:
            node.set_reliability(original_reliability)
    
    def get_discounted_spdf(self, node_name: str, hypothesis: Optional[Set] = None,
                           reliability: Optional[float] = None) -> float:
        """
        计算节点经过可靠性折扣后的BPA对假设的类概率函数值
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]
        if reliability is None:
            reliability = node.reliability
        
        # 临时修改可靠性计算，然后恢复
        original_reliability = node.reliability
        try:
            node.set_reliability(reliability)
            return node.get_spdf(hypothesis)
        finally:
            node.set_reliability(original_reliability)
    
    def get_propagated_belief(self, node_name: str, hypothesis: Optional[Set] = None) -> float:
        """
        计算节点经过推理可靠性传播后的置信度
        """
        reliability_result = self.calculate_node_reliability(node_name)
        return self.get_discounted_belief(node_name, hypothesis, reliability_result.reliability)
    
    def get_propagated_plausibility(self, node_name: str, hypothesis: Optional[Set] = None) -> float:
        """
        计算节点经过推理可靠性传播后的似然度
        """
        reliability_result = self.calculate_node_reliability(node_name)
        return self.get_discounted_plausibility(node_name, hypothesis, reliability_result.reliability)
    
    def get_propagated_spdf(self, node_name: str, hypothesis: Optional[Set] = None) -> float:
        """
        计算节点经过推理可靠性传播后的类概率函数值
        """
        reliability_result = self.calculate_node_reliability(node_name)
        return self.get_discounted_spdf(node_name, hypothesis, reliability_result.reliability)
    
    def calculate_node_reliability(self, node_name: str) -> ReliabilityResult:
        """
        计算节点的推理可靠性，根据节点类型、输入边的权重和父节点推理时的可靠性
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
        
        # 检查缓存
        if node_name in self._reliability_cache:
            return self._reliability_cache[node_name]
            
        node = self.nodes[node_name]
        
        # 获取所有父节点（输入节点）
        parent_nodes = node.get_parent_nodes(self.graph)
        
        # 如果没有父节点，直接返回节点自身的可靠性
        if not parent_nodes:
            result = ReliabilityResult(node.reliability, node.reliability)
            self._reliability_cache[node_name] = result
            return result
        
        # 计算每个父节点的推理可靠性
        parent_reliabilities = {}
        for parent in parent_nodes:
            parent_result = self.calculate_node_reliability(parent)
            parent_reliabilities[parent] = parent_result.certification
        
        # 获取所有输入边的权重
        input_weights = {}
        for parent in parent_nodes:
            weight = self.edge_weights.get((parent, node_name), 1)
            input_weights[parent] = weight
        
        # 根据节点类型组合计算结果
        if node.node_type == NodeType.AND:
            result = self._calculate_and_reliability(node, parent_nodes, parent_reliabilities, input_weights)
        elif node.node_type == NodeType.OR:
            result = self._calculate_or_reliability(node, parent_nodes, parent_reliabilities, input_weights)
        elif node.node_type == NodeType.FUSION:
            result = self._calculate_fusion_reliability(node, parent_nodes, parent_reliabilities, input_weights)
        else:
            result = ReliabilityResult(node.reliability, node.reliability)
        
        self._reliability_cache[node_name] = result
        return result
    
    def _calculate_and_reliability(self, node: EvidenceNode, parent_nodes: List[str],
                                 parent_reliabilities: Dict[str, float], input_weights: Dict[str, float]) -> ReliabilityResult:
        """计算AND节点的可靠性"""
        weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
        combined_value = min(weighted_reliabilities) if weighted_reliabilities else 0
        propagated_reliability = combined_value * node.reliability

        if not node.has_bpa():
            certification = propagated_reliability
        else:
            certification = self.get_discounted_spdf(node.name, reliability=propagated_reliability)
        
        return ReliabilityResult(certification, propagated_reliability)
    
    def _calculate_or_reliability(self, node: EvidenceNode, parent_nodes: List[str],
                                parent_reliabilities: Dict[str, float], input_weights: Dict[str, float]) -> ReliabilityResult:
        """计算OR节点的可靠性"""
        weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
        combined_value = max(weighted_reliabilities) if weighted_reliabilities else 0
        propagated_reliability = combined_value * node.reliability

        if not node.has_bpa():
            certification = propagated_reliability
        else:
            certification = self.get_discounted_spdf(node.name, reliability=propagated_reliability)
        
        return ReliabilityResult(certification, propagated_reliability)
    
    def _calculate_fusion_reliability(self, node: EvidenceNode, parent_nodes: List[str],
                                    parent_reliabilities: Dict[str, float], input_weights: Dict[str, float]) -> ReliabilityResult:
        """计算FUSION节点的可靠性"""
        try:
            # 获取所有输入BPA（考虑可靠性）
            input_bpas = []
            for parent in parent_nodes:
                if parent in node.input_bpas:
                    # 使用FUSION节点存储的输入BPA和可靠性
                    input_data = node.input_bpas[parent]
                    input_bpa = input_data.bpa
                    input_reliability = input_data.reliability * parent_reliabilities[parent] * input_weights[parent]
                    
                    # 对输入BPA进行可靠性折扣
                    discounted_bpa = self.discount_bpa(input_bpa, input_reliability)
                    input_bpas.append(discounted_bpa)
                else:
                    # 使用前驱节点自身的BPA，并考虑其可靠性
                    pred_bpa = self.nodes[parent].bpa
                    pred_reliability = parent_reliabilities[parent] * input_weights[parent]
                    
                    # 对前驱节点BPA进行可靠性折扣             
                    discounted_bpa = self.discount_bpa(pred_bpa, pred_reliability)
                    input_bpas.append(discounted_bpa)
            
            # 逐步融合所有输入BPA
            if not input_bpas:
                return ReliabilityResult(node.reliability, node.reliability)
                
            combined_bpa = input_bpas[0].copy()
            for i in range(1, len(input_bpas)):
                combined_bpa = EvidenceNode.merge_evidence_dempster(
                    combined_bpa, input_bpas[i], node.frame)
            
            # 保存原始BPA，设置新的BPA(对于fusion类型而言，不在保存原始的bpa)
            # original_bpa = node.bpa
            # try:
            node.set_bpa(combined_bpa)
            SPDF = node.get_spdf()
            certification = SPDF * node.reliability
                
            return ReliabilityResult(certification, node.reliability)
            # finally:
            #     # 恢复原始BPA
            #     node.set_bpa(original_bpa)
                
        except Exception as e:
            # 如果融合失败，使用加权平均作为备选方案
            print(f"FUSION节点 {node.name} 可靠性计算失败: {e}, 使用加权平均")
            weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
            combined_value = sum(weighted_reliabilities) / len(weighted_reliabilities) if weighted_reliabilities else 0
            return ReliabilityResult(combined_value * node.reliability, node.reliability)
    
    def propagate_certification(self, node_name: str, hypothesis: Optional[Set] = None) -> float:
        """
        从起始节点传播证据，计算目标节点的置信度和似然度
        """
        if node_name not in self.nodes:
            raise ValueError("计算的节点不存在")
              
        return self.get_propagated_spdf(node_name, hypothesis)
    
    def has_cycle(self) -> bool:
        """
        检查图中是否有环
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True
    
    def get_node_info(self, node_name: str) -> str:
        """获取节点的详细信息"""
        if node_name not in self.nodes:
            return f"节点 {node_name} 不存在"
        
        node = self.nodes[node_name]
        return str(node)
    
    def get_graph_info(self) -> Dict[str, Any]:
        """获取图的统计信息"""
        return {
            "节点数量": len(self.nodes),
            "边数量": len(self.edge_weights),
            "是否有环": self.has_cycle(),
            "节点类型分布": {
                node_type.name: sum(1 for node in self.nodes.values() if node.node_type == node_type)
                for node_type in NodeType
            }
        }
    
    def __str__(self) -> str:
        """返回图的字符串表示"""
        if not self.graph:
            return "空图"
            
        result = ["有向无环图结构:"]
        for u in self.graph:
            neighbors = [f"{v}({self.edge_weights.get((u, v), 1)})" for v in self.graph[u]]
            result.append(f"{u} -> {', '.join(neighbors) if neighbors else '(无出边)'}")
        
        result.append("\n节点证据信息:")
        for node_name, node in self.nodes.items():
            result.append(str(node))
            
        return "\n".join(result)