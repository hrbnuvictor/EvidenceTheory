"""
带证据推理功能的加权有向无环图（Evidence Reasoning DAG）实现

该模块实现了支持证据推理的有向无环图，包括：
1. 三种节点类型：AND、OR、FUSION
2. 证据推理功能：辨识框架、基本概率分配（BPA）
3. 节点可靠性计算
4. 证据传播和融合
5. 拓扑排序和路径计算

作者：Alexander with AI Assistant
日期：2025年9月19日
版本：1.5
"""

from collections import deque, defaultdict
from enum import Enum

class NodeType(Enum):
    """节点类型枚举类"""
    AND = 1      # AND类型节点：所有输入必须满足
    OR = 2       # OR类型节点：任一输入满足即可
    FUSION = 3   # FUSION类型节点：使用Dempster规则融合输入证据


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
        其中，∅代表空集，Ω代表全集
    """
    
    def __init__(self, name, frame=None, bpa=None, dimension = 20, reliability=1.0, node_type=NodeType.AND):
        """
        初始化证据节点
        
        参数:
            name (str): 节点名称
            frame (set, optional): 辨识框架. 默认为空集
            bpa (dict, optional): 基本概率分配. 默认为空字典
            reliability (float, optional): 节点可靠性. 默认为1.0
            node_type (NodeType, optional): 节点类型. 默认为NodeType.AND
        """
        self.name = name
        self.frame = frame if frame else set()  # 辨识框架
        self.input_bpas = {}  # 存储每个输入节点的BPA和可靠性（仅对FUSION节点有效）
        
        if bpa is not None:         # 基本概率分配
            self.set_bpa(bpa)
        else:
            self.bpa = {}  
        self.set_reliability(reliability)  # 节点可靠性，默认值为1
        self.node_type = node_type  # 节点类型
        self.dimension = dimension  # 节点客体的辨识框架中元素的个数
                
    def set_frame(self, frame):
        """设置辨识框架"""
        self.frame = set(frame)
        
    def set_bpa(self, bpa):
        """
        设置基本概率分配
        
        参数:
            bpa (dict): 基本概率分配字典
            
        异常:
            ValueError: 当BPA值之和小于等于1或子集不在辨识框架内时抛出
        """
        # 验证BPA值是否有效
        total = 0
        for subset, mass in bpa.items():
            if mass < 0:
                raise ValueError(f"BPA值不能小于0，子集 {subset} 的值为 {mass}")
            total += mass
            
        if total > 1.0 + 1e-10:  # 考虑浮点数精度
            raise ValueError(f"BPA值之和不能大于1，当前总和为 {total:.6f}")
            
        # 验证所有子集是否都在辨识框架内
        for subset in bpa.keys():
            if not subset.issubset(self.frame):
                if (subset != frozenset({'∅'})) and (subset != frozenset({'Ω'})):
                    raise ValueError(f"子集 {subset} 不在辨识框架内")
                
        self.bpa = bpa
        
    def set_input_bpa(self, input_node, bpa, reliability=1.0):
        """
        为FUSION节点设置输入节点的BPA及其可靠性
        
        参数:
            input_node (str): 输入节点名称
            bpa (dict): 基本概率分配字典
            reliability (float, optional): 该输入BPA的可靠性. 默认为1.0
            
        异常:
            ValueError: 当节点不是FUSION类型时抛出
        """
        if self.node_type != NodeType.FUSION:
            raise ValueError("只有FUSION类型的节点可以设置输入BPA")
            
        # 验证BPA值是否有效
        total = 0
        for subset, mass in bpa.items():
            if mass < 0:
                raise ValueError(f"BPA值不能小于0，子集 {subset} 的值为 {mass}")
            total += mass
            
        if total > 1.0 + 1e-10:  # 考虑浮点数精度
            raise ValueError(f"BPA值之和不能大于1，当前总和为 {total:.6f}")
            
        # 验证所有子集是否都在辨识框架内
        for subset in bpa.keys():
            if not subset.issubset(self.frame):
                if (subset != frozenset({'∅'})) and (subset != frozenset({'Ω'})):
                    raise ValueError(f"子集 {subset} 不在辨识框架内")
                
        # 验证可靠性值是否有效
        if reliability < 0 or reliability > 1:
            raise ValueError("可靠性必须在0到1之间")
                
        self.input_bpas[input_node] = {
            'bpa': bpa,
            'reliability': reliability
        }
        
    def set_reliability(self, reliability):
        """
        设置节点可靠性
        
        参数:
            reliability (float): 可靠性值，范围[0,1]
            
        异常:
            ValueError: 当可靠性值不在[0,1]范围内时抛出
        """
        if reliability < 0 or reliability > 1:
            raise ValueError("可靠性必须在0到1之间")
        self.reliability = reliability
        
    def set_node_type(self, node_type):
        """设置节点类型"""
        self.node_type = node_type

    def set_dimension(self, dimension):
        """设置节点维数"""
        self.dimension = dimension

    def get_belief(self, hypothesis=None):
        """
        计算假设的置信函数值
        
        参数:
            hypothesis (set): 要评估的假设
            
        返回:
            float: 假设的置信度
            
        异常:
            ValueError: 当假设不在辨识框架内时抛出
        """
        if hypothesis is None:
            hypothesis = self.frame

        if not hypothesis.issubset(self.frame):
            if hypothesis != frozenset({'Ω'}) and hypothesis != frozenset({'∅'}):
                raise ValueError("假设必须在辨识框架内")
            
        discounted_bpa = EvidenceNode.correct_bpa(self.bpa,self.reliability) 

        belief = 0.0
        for subset, mass in discounted_bpa.items():
            if subset.issubset(hypothesis) or (hypothesis == frozenset({'Ω'}) and subset != frozenset({'∅'})):
                belief += mass
        return belief

    def get_plausibility(self, hypothesis=None):
        """
        计算假设的似然函数值
        
        参数:
            hypothesis (set): 要评估的假设
            
        返回:
            float: 假设的似然度
            
        异常:
            ValueError: 当假设不在辨识框架内时抛出
        """
        if hypothesis is None:
            hypothesis = self.frame

        if not hypothesis.issubset(self.frame):
            if hypothesis != frozenset({'Ω'}) and hypothesis != frozenset({'∅'}):
                raise ValueError("假设必须在辨识框架内")
            
        discounted_bpa = EvidenceNode.correct_bpa(self.bpa,self.reliability)     
        plausibility = 0.0
        for subset, mass in discounted_bpa.items():
            if subset == frozenset({'Ω'}):
                if hypothesis != frozenset({'∅'}):
                    plausibility += mass
            else:
                if subset.intersection(hypothesis) or (hypothesis == frozenset({'Ω'}) and subset != frozenset({'∅'})):
                    plausibility += mass
        return plausibility

    def get_spdf(self, hypothesis=None) :
        """
        计算假设的类概率函数值
        
        参数:
            hypothesis (set): 要评估的假设
            dimension: 辨识框架的维度
            
        返回:
            float: 假设的类概率函数值
            
        异常:
            ValueError: 当假设不在辨识框架内时抛出
        """
        if hypothesis is None:
            hypothesis = self.frame

        if not hypothesis.issubset(self.frame):
            if hypothesis != frozenset({'Ω'}) and hypothesis != frozenset({'∅'}):
                raise ValueError("假设必须在辨识框架内")
        
        count = 0
        # 查看假设中含有辨识框架元素的个数。
        for subset in self.frame:
            if {subset}.issubset(hypothesis) or hypothesis == frozenset({'Ω'}):
                count += 1
                
        spdf = self.get_belief(hypothesis) + (self.get_plausibility(hypothesis) - self.get_belief(hypothesis)) * count / self.dimension
        return spdf
    
    def correct_bpa(bpa, reliability):
        """
        根据可靠性对BPA进行折扣
        
        参数:
            bpa (dict): 原始BPA
            reliability (float): 可靠性值
            
        返回:
            dict: 折扣后的BPA
        """
        if reliability < 0 or reliability > 1:
            raise ValueError("可靠性必须在0到1之间")
            
        discounted_bpa = {}
        masssum = 0.0        
        for subset, mass in bpa.items():
            discounted_bpa[subset] = mass * reliability
            masssum += discounted_bpa[subset]

        # 将剩余的概率质量分配给全集（表示不确定性）         
        frame_set = frozenset({'Ω'})
        if frame_set in discounted_bpa:
            discounted_bpa[frame_set] += 1 - masssum
        else:
            discounted_bpa[frame_set] = 1 - masssum
       
        return discounted_bpa
    
    def merge_evidence_dempster(bpa1, bpa2,frame):
        """
        使用Dempster规则合成两个bpa
        
        参数:
            ev1 (str): 第一个证据
            ev2 (str): 第二个证据
            
        返回:
            dict: 合成后的BPA
            
        异常:
            ValueError: 当节点不存在、辨识框架不一致或证据完全冲突时抛出
        """
        for subset in bpa1.keys():
            if not subset.issubset(frame):
                if (subset != frozenset({'∅'})) and (subset != frozenset({'Ω'})):
                    raise ValueError(f"子集 {subset} 不在辨识框架内")
        for subset in bpa2.keys():
            if not subset.issubset(frame):
                if (subset != frozenset({'∅'})) and (subset != frozenset({'Ω'})):
                    raise ValueError(f"子集 {subset} 不在辨识框架内")
        # 计算冲突系数K
        K = 0
        for A in bpa1:
            for B in bpa2:
                if A == frozenset({'∅'}) or B == frozenset({'∅'}):
                    if not A.intersection(B):
                        K += bpa1[A] * bpa2[B]
                else:
                    if not A.intersection(B) and A != frozenset({'Ω'}) and B != frozenset({'Ω'}):
                        K += bpa1[A] * bpa2[B]
        
        # 如果冲突系数为1，无法合成
        if abs(K - 1) < 1e-10:
            raise ValueError("证据完全冲突，无法合成")
        
        # 计算合成后的BPA
        combined_bpa = {}
        for A in bpa1:
            for B in bpa2:
                if A == frozenset({'Ω'}) and B != frozenset({'∅'}):
                    intersection = B
                else:
                    if B == frozenset({'Ω'}) and A != frozenset({'∅'}):
                        intersection = A
                    else:
                        intersection = A.intersection(B)
                if intersection:
                    if intersection not in combined_bpa:
                        combined_bpa[intersection] = 0
                    combined_bpa[intersection] += bpa1[A] * bpa2[B] / (1 - K)
        
        # 归一化
        total = sum(combined_bpa.values())
        if total > 1:
            for subset in combined_bpa:
                combined_bpa[subset] /= total
        else:
            frame_set = frozenset({'Ω'})            
            if frame_set in combined_bpa:
                combined_bpa[frame_set] += 1 - total
            else:
                combined_bpa[frame_set] = 1 - total
        return combined_bpa

    def __str__(self):
        """返回节点的字符串表示"""
        if self.node_type == NodeType.FUSION:
            input_info = []
            for input_node, data in self.input_bpas.items():
                input_info.append(f"{input_node}(可靠性={data['reliability']:.2f})")
            return f"节点 {self.name}: 类型={self.node_type.name}, 可靠性={self.reliability:.2f}, 辨识框架={self.frame}, 输入BPA数量={len(self.input_bpas)} [{', '.join(input_info)}]"
        else:
            return f"节点 {self.name}: 类型={self.node_type.name}, 可靠性={self.reliability:.2f}, 辨识框架={self.frame}, BPA={self.bpa}"

class EvidenceDAG:
    """
    带证据推理功能的有向无环图类
    
    属性:
        graph (defaultdict): 图的邻接表表示
        in_degree (defaultdict): 每个节点的入度
        edge_weights (dict): 边的权重字典，键为(源节点,目标节点)元组
        nodes (dict): 证据节点字典，键为节点名称，值为EvidenceNode对象
    """
    
    def __init__(self, edges=None):
        """
        初始化带证据推理功能的有向无环图
        
        参数:
            edges (list, optional): 可选的边列表，格式为 [(源节点, 目标节点, 权重), ...]
        """
        # 使用字典存储图的邻接表表示，值为(目标节点, 权重)的元组列表
        self.graph = defaultdict(list)
        # 存储每个节点的入度
        self.in_degree = defaultdict(int)
        # 存储所有边的权重
        self.edge_weights = {}
        # 存储所有证据节点
        self.nodes = {}
        
        # 如果提供了边列表，则初始化图
        if edges is not None:
            for u, v, weight in edges:
                self.add_edge(u, v, weight)
    
    def add_node(self, node_name, frame=None, bpa=None, dimension = 20, reliability=1.0, node_type=NodeType.AND):
        """
        添加证据节点
        
        参数:
            node_name (str): 节点名称
            frame (set, optional): 辨识框架. 默认为None
            bpa (dict, optional): 基本概率分配. 默认为None
            reliability (float, optional): 节点可靠性. 默认为1.0
            node_type (NodeType, optional): 节点类型. 默认为NodeType.AND
            
        返回:
            bool: 如果节点添加成功返回True，否则返回False
        """
        if node_name not in self.nodes:
            self.nodes[node_name] = EvidenceNode(node_name, frame, bpa, dimension, reliability, node_type)
            self.graph[node_name] = []
            self.in_degree[node_name] = 0
            return True
        return False
    
    def add_edge(self, u, v, weight=1):
        """
        添加从u到v的有向边，带有权重
        
        参数:
            u (str): 源节点名称
            v (str): 目标节点名称
            weight (float, optional): 边权重. 默认为1
            
        返回:
            bool: 如果边添加成功返回True，如果添加会导致环则返回False
        """
        # 确保节点存在
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)
        
        # 检查添加这条边是否会创建环
        if self.would_create_cycle(u, v):
            return False
        
        # 添加边
        self.graph[u].append(v)
        # 存储边的权重
        self.edge_weights[(u, v)] = weight
        # 增加目标节点的入度
        self.in_degree[v] += 1
        return True
    
    def update_edge_weight(self, u, v, weight):
        """
        更新边的权重
        
        参数:
            u (str): 源节点名称
            v (str): 目标节点名称
            weight (float): 新的权重值
            
        返回:
            bool: 如果更新成功返回True，否则返回False
        """
        if (u, v) in self.edge_weights:
            self.edge_weights[(u, v)] = weight
            return True
        return False
    
    def get_edge_weight(self, u, v):
        """
        获取边的权重
        
        参数:
            u (str): 源节点名称
            v (str): 目标节点名称
            
        返回:
            float: 边的权重，如果边不存在则返回None
        """
        return self.edge_weights.get((u, v), None)
    
    def would_create_cycle(self, u, v):
        """
        检查添加从u到v的边是否会创建环
        
        参数:
            u (str): 源节点名称
            v (str): 目标节点名称
            
        返回:
            bool: 如果会创建环返回True，否则返回False
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
    
    def remove_edge(self, u, v):
        """
        移除从u到v的边
        
        参数:
            u (str): 源节点名称
            v (str): 目标节点名称
            
        返回:
            bool: 如果边移除成功返回True，否则返回False
        """
        if u in self.graph and v in self.graph[u]:
            self.graph[u].remove(v)
            self.in_degree[v] -= 1
            # 移除边的权重
            if (u, v) in self.edge_weights:
                del self.edge_weights[(u, v)]
            return True
        return False
    
    def get_nodes(self):
        """
        返回图中所有节点
        
        返回:
            list: 图中所有节点的名称列表
        """
        return list(self.graph.keys())
    
    def get_edges(self):
        """
        返回图中所有边及其权重
        
        返回:
            list: 图中所有边的列表，格式为[(源节点, 目标节点, 权重), ...]
        """
        edges = []
        for u in self.graph:
            for v in self.graph[u]:
                weight = self.edge_weights.get((u, v), 1)
                edges.append((u, v, weight))
        return edges
    
    def topological_sort(self):
        """
        使用Kahn算法进行拓扑排序
        
        返回:
            list: 拓扑排序后的节点列表
            
        异常:
            ValueError: 当图中存在环时抛出
        """
        # 复制入度字典
        in_degree = self.in_degree.copy()
        
        # 收集所有入度为0的节点
        queue = deque([node for node in self.graph if in_degree[node] == 0])
        topo_order = []
        
        while queue:
            u = queue.popleft()
            topo_order.append(u)
            
            # 减少所有邻居节点的入度
            for v in self.graph[u]:
                in_degree[v] -= 1
                # 如果邻居节点入度变为0，加入队列
                if in_degree[v] == 0:
                    queue.append(v)
        
        # 检查是否所有节点都被访问（判断是否有环）
        if len(topo_order) != len(self.graph):
            raise ValueError("图中存在环，无法进行拓扑排序")
        
        return topo_order
    
    def longest_path(self, start, end):
        """
        计算从start到end的最长路径（关键路径）及其权重
        
        参数:
            start (str): 起始节点名称
            end (str): 终止节点名称
            
        返回:
            tuple: (路径列表, 路径总权重)，如果没有路径则返回(None, -∞)
        """
        # 获取拓扑排序
        topo_order = self.topological_sort()
        
        # 初始化距离字典
        dist = {node: float('-inf') for node in self.graph}
        dist[start] = 0
        
        # 存储前驱节点，用于重建路径
        prev = {}
        
        # 按照拓扑顺序处理每个节点
        for node in topo_order:
            if dist[node] != float('-inf'):
                for neighbor in self.graph[node]:
                    weight = self.edge_weights.get((node, neighbor), 1)
                    if dist[neighbor] < dist[node] + weight:
                        dist[neighbor] = dist[node] + weight
                        prev[neighbor] = node
        
        # 重建路径
        path = []
        current = end
        while current != start:
            path.append(current)
            if current not in prev:
                return None, float('-inf')  # 没有路径
            current = prev[current]
        path.append(start)
        path.reverse()
        
        return path, dist[end]
    
    def shortest_path(self, start, end):
        """
        计算从start到end的最短路径及其权重
        
        参数:
            start (str): 起始节点名称
            end (str): 终止节点名称
            
        返回:
            tuple: (路径列表, 路径总权重)，如果没有路径则返回(None, ∞)
        """
        # 获取拓扑排序
        topo_order = self.topological_sort()
        
        # 初始化距离字典
        dist = {node: float('inf') for node in self.graph}
        dist[start] = 0
        
        # 存储前驱节点，用于重建路径
        prev = {}
        
        # 按照拓扑顺序处理每个节点
        for node in topo_order:
            if dist[node] != float('inf'):
                for neighbor in self.graph[node]:
                    weight = self.edge_weights.get((node, neighbor), 1)
                    if dist[neighbor] > dist[node] + weight:
                        dist[neighbor] = dist[node] + weight
                        prev[neighbor] = node
        
        # 重建路径
        path = []
        current = end
        while current != start:
            path.append(current)
            if current not in prev:
                return None, float('inf')  # 没有路径
            current = prev[current]
        path.append(start)
        path.reverse()
        
        return path, dist[end]
    
    def combine_evidence_dempster(self, node1, node2):
        """
        使用Dempster规则合成两个节点的证据
        
        参数:
            node1 (str): 第一个节点名称
            node2 (str): 第二个节点名称
            
        返回:
            dict: 合成后的BPA
            
        异常:
            ValueError: 当节点不存在、辨识框架不一致或证据完全冲突时抛出
        """
        if node1 not in self.nodes or node2 not in self.nodes:
            raise ValueError("节点不存在")
            
        # 获取两个节点的evidence
        ev1 = self.nodes[node1]
        ev2 = self.nodes[node2]
        if ev1.frame != ev2.frame:
            raise ValueError("两个节点的辨识框架不一致")
        combined_bpa = EvidenceNode.merge_evidence_dempster(ev1.bpa,ev2.bpa,ev1.frame)            
        return combined_bpa
    
    def discount_bpa(self, bpa, reliability):
        """
        根据可靠性对BPA进行折扣
        
        参数:
            bpa (dict): 原始BPA
            reliability (float): 可靠性值
            
        返回:
            dict: 折扣后的BPA
        """
        if reliability < 0 or reliability > 1:
            raise ValueError("可靠性必须在0到1之间")
            
        discounted_bpa = {}
        masssum = 0.0        
        for subset, mass in bpa.items():
            discounted_bpa[subset] = mass * reliability
            masssum += discounted_bpa[subset]

        # 将剩余的概率质量分配给全集（表示不确定性）         
        frame_set = frozenset({'Ω'})
        if frame_set in discounted_bpa:
            discounted_bpa[frame_set] += 1 - masssum
        else:
            discounted_bpa[frame_set] = 1 - masssum
       
        return discounted_bpa
    
    def get_discounted_belief(self, node_name, hypothesis=None, reliability=None):
        """
        计算节点经过可靠性折扣后的BPA对假设的置信度
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            reliability (float, optional): 指定的可靠性值，如果为None则使用节点可靠性
            
        返回:
            float: 折扣后的置信度
            
        异常:
            ValueError: 当节点不存在或假设不在辨识框架内时抛出
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
        # 使用指定的可靠性或节点自身的可靠性
        node = self.nodes[node_name]
        if reliability is None:
            reliability = node.reliability
        
        reliabilitybak = node.reliability
        node.set_reliability(reliability)
        belief = node.get_belief(hypothesis)
        node.set_reliability(reliabilitybak)
        return belief
    
    def get_discounted_plausibility(self, node_name, hypothesis=None, reliability=None):
        """
        计算节点经过可靠性折扣后的BPA对假设的似然度
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            reliability (float, optional): 指定的可靠性值，如果为None则使用节点可靠性
            
        返回:
            float: 折扣后的似然度
            
        异常:
            ValueError: 当节点不存在或假设不在辨识框架内时抛出
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]

        if hypothesis is None:
            hypothesis = node.frame

        if not hypothesis.issubset(node.frame):
            raise ValueError("假设必须在辨识框架内")
        
        # 使用指定的可靠性或节点自身的可靠性
        if reliability is None:
            reliability = node.reliability
            
        reliabilitybak = node.reliability
        node.set_reliability(reliability)
        plausibility = node.get_plausibility(hypothesis)
        node.set_reliability(reliabilitybak)                
        return plausibility
    
    def get_discounted_spdf(self, node_name, hypothesis=None, reliability=None):
        """
        计算节点经过可靠性折扣后的BPA对假设的类概率函数值
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            dimension: 辨识框架的维度
            reliability (float, optional): 指定的可靠性值，如果为None则使用节点可靠性
            
        返回:
            float: 折扣后的类概率函数值
            
        异常:
            ValueError: 当节点不存在或假设不在辨识框架内时抛出
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]
            # 使用指定的可靠性或节点自身的可靠性
        if reliability is None:
            reliability = node.reliability

        reliabilitybak = node.reliability
        node.set_reliability(reliability)
        spdf = node.get_spdf(hypothesis)
        node.set_reliability(reliabilitybak) 
        # 计算类概率函数
        return spdf
    
    def get_propagated_belief(self, node_name, hypothesis=None):
        """
        计算节点经过推理可靠性传播后的置信度
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            
        返回:
            float: 传播后的置信度
        """
        [certification, reliability] = self.calculate_node_reliability(node_name)
        return self.get_discounted_belief(node_name, hypothesis, reliability)
    
    def get_propagated_plausibility(self, node_name, hypothesis=None):
        """
        计算节点经过推理可靠性传播后的似然度
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            
        返回:
            float: 传播后的似然度
        """      
        [certification, reliability] = self.calculate_node_reliability(node_name)
        return self.get_discounted_plausibility(node_name, hypothesis, reliability)
    
    def get_propagated_spdf(self, node_name, hypothesis=None):
        """
        计算节点经过推理可靠性传播后的类概率函数值
        
        参数:
            node_name (str): 节点名称
            hypothesis (set): 要评估的假设
            dimension: 辨识框架的维度
            
        返回:
            float: 传播后的类概率函数值
        """
        [certification,reliability] = self.calculate_node_reliability(node_name)
        return self.get_discounted_spdf(node_name, hypothesis, reliability)
        
    def calculate_node_reliability(self, node_name):
        """
        计算节点的推理可靠性，根据节点类型、输入边的权重和父节点推理时的可靠性
        
        对于FUSION节点，可靠性基于输入BPA经过可靠性折扣和Dempster合成后的结果
        
        参数:
            node_name (str): 节点名称
            
        返回:
            float: 计算后的推理可靠性值
            
        异常:
            ValueError: 当节点不存在时抛出
        """
        if node_name not in self.nodes:
            raise ValueError("节点不存在")
            
        node = self.nodes[node_name]
        
        # 获取所有父节点（输入节点）
        parent_nodes = []
        for u in self.graph:
            if node_name in self.graph[u]:
                parent_nodes.append(u)
        
        # 如果没有父节点，直接返回节点自身的可靠性
        if not parent_nodes:
            return [node.reliability,node.reliability]
        
        # 计算每个父节点的推理可靠性
        parent_reliabilities = {}
        for parent in parent_nodes:
            [parent_reliabilities[parent], reliabilitynone] = self.calculate_node_reliability(parent)
        
        # 获取所有输入边的权重
        input_weights = {}
        for parent in parent_nodes:
            weight = self.edge_weights.get((parent, node_name), 1)
            input_weights[parent] = weight
        
        # 根据节点类型组合计算结果
        if node.node_type == NodeType.AND:
            # AND节点：取所有加权值的最小值
            weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
            combined_value = min(weighted_reliabilities) if weighted_reliabilities else 0
            propagated_reliability = combined_value * node.reliability

            if node.bpa == {}:
                certification = propagated_reliability
            else:
                certification = self.get_discounted_spdf(node_name,reliability=combined_value * node.reliability)
            # 最终可靠性为组合值乘以节点自身的可靠性
            return [certification,propagated_reliability]
            
        elif node.node_type == NodeType.OR:
            # OR节点：取所有加权值的最大值
            weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
            combined_value = max(weighted_reliabilities) if weighted_reliabilities else 0
            propagated_reliability = combined_value * node.reliability

            if node.bpa == {}:
                certification = propagated_reliability
            else:
                certification = self.get_discounted_spdf(node_name,reliability=combined_value * node.reliability)
            # 最终可靠性为组合值乘以节点自身的可靠性
            return [certification,propagated_reliability]
            
        elif node.node_type == NodeType.FUSION:
            # FUSION节点：基于输入BPA经过可靠性折扣和Dempster合成后的结果
            try:
                # 获取所有输入BPA（考虑可靠性）
                input_bpas = []
                for parent in parent_nodes:
                    if parent in node.input_bpas:
                        # 使用FUSION节点存储的输入BPA和可靠性
                        input_data = node.input_bpas[parent]
                        input_bpa = input_data['bpa']
                        input_reliability = input_data['reliability'] * parent_reliabilities[parent] * input_weights[parent]
                        
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
                    return node.reliability
                    
                combined_bpa = input_bpas[0].copy()
                for i in range(1, len(input_bpas)):
                    combined_bpa = EvidenceNode.merge_evidence_dempster(
                        combined_bpa, input_bpas[i], node.frame)
                node.set_bpa(combined_bpa)
                SPDF = node.get_spdf()
                certification = SPDF * node.reliability

                return [certification, node.reliability]
                
            except Exception as e:
                # 如果融合失败，使用加权平均作为备选方案
                print(f"FUSION节点 {node_name} 可靠性计算失败: {e}, 使用加权平均")
                weighted_reliabilities = [input_weights[parent] * parent_reliabilities[parent] for parent in parent_nodes]
                combined_value = sum(weighted_reliabilities) / len(weighted_reliabilities) if weighted_reliabilities else 0
                return [combined_value * node.reliability,node.reliability]
    
    def propagate_certification(self, current_nodename, hypothesis=None):
        """
        从起始节点传播证据，计算目标节点的置信度和似然度
        
        参数:
            start_node (str): 起始节点名称
            hypothesis (set): 要评估的假设
            
        返回:
            dict: 包含每个节点对假设的置信度、似然度和计算后的可靠性
            
        异常:
            ValueError: 当起始节点不存在或假设不在辨识框架内时抛出
        """
        if current_nodename not in self.nodes:
            raise ValueError("计算的节点不存在")
        # if hypothesis is None:
        #     hypothesis = self.nodes[current_nodename].frame

        # if not hypothesis.issubset(self.nodes[current_nodename].frame):
        #     if hypothesis != frozenset({'Ω'}) and hypothesis != frozenset({'∅'}):
        #         raise ValueError("假设不在计算节点的辨识框架内")
              
        SPDF = self.get_propagated_spdf(current_nodename,hypothesis)
        return SPDF
    
    def has_cycle(self):
        """
        检查图中是否有环
        
        返回:
            bool: 如果图中存在环返回True，否则返回False
        """
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True
    
    def __str__(self):
        """返回图的字符串表示"""
        if not self.graph:
            return "空图"
            
        result = ["有向无环图结构:"]
        for u in self.graph:
            if self.graph[u]:
                neighbors = []
                for v in self.graph[u]:
                    weight = self.edge_weights.get((u, v), 1)
                    neighbors.append(f"{v}({weight})")
                result.append(f"{u} -> {', '.join(neighbors)}")
            else:
                result.append(f"{u} -> (无出边)")
        
        result.append("\n节点证据信息:")
        for node_name, node in self.nodes.items():
            result.append(str(node))
            if node.node_type == NodeType.FUSION and node.input_bpas:
                for input_node, data in node.input_bpas.items():
                    result.append(f"  输入节点 {input_node}: 可靠性={data['reliability']:.2f}, BPA={data['bpa']}")
            
        return "\n".join(result)