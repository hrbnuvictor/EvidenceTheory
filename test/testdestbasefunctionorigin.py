import sys
import os
import platform

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules import destbasefunctionorigin

# 示例用法
if __name__ == "__main__":
    # 创建证据DAG实例
    dag = destbasefunctionorigin.EvidenceDAG()
    
    # 添加节点和证据信息
    dag.add_node('E1', frame={'a', 'b', 'c'}, reliability=0.8, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['E1'].set_bpa({
        frozenset({'a'}): 0.6,
        frozenset({'b'}): 0.3,
        frozenset({'c'}): 0.1
    })
    
    dag.add_node('A', frame={'a1', 'a2'}, reliability=1, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['A'].set_bpa({
        frozenset({'a1'}): 0.3,
        frozenset({'a2'}): 0.5
    })
    
    dag.add_node('E2', frame={'a', 'b', 'c'}, reliability=0.6, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['E2'].set_bpa({
        frozenset({'a'}): 0.5,
        frozenset({'b'}): 0.3,
        frozenset({'c'}): 0.2
    })
        
    dag.add_node('E3', frame={'a', 'b'}, reliability=0.9, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['E3'].set_bpa({
        frozenset({'a'}): 0.6,
        frozenset({'b'}): 0.4
    })
    
    dag.add_node('E4', frame={'a', 'b'}, reliability=0.5, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['E4'].set_bpa({
        frozenset({'a'}): 0.3,
        frozenset({'b'}): 0.7
    })
    
    dag.add_node('InterMedia', frame={'a', 'b'}, reliability=1, node_type=destbasefunctionorigin.NodeType.OR)
    # dag.nodes['InterMedia'].set_bpa({
    #     frozenset({'a'}): 0.31,
    #     frozenset({'b'}): 0.69
    # })

    dag.add_node('N', frame={'n'}, reliability=1, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['N'].set_bpa({
        frozenset({'n'}): 0.7
    })
    
    dag.add_node('E5', frame={'a', 'b'}, reliability=0.7, node_type=destbasefunctionorigin.NodeType.AND)
    dag.nodes['E5'].set_bpa({
        frozenset({'a'}): 0.5,
        frozenset({'b'}): 0.5
    })
    
    # 创建FUSION节点
    dag.add_node('H', frame={'h1', 'h2', 'h3'}, reliability=1, node_type=destbasefunctionorigin.NodeType.FUSION)
    
    # 为FUSION节点设置输入BPA及其可靠性
    dag.nodes['H'].set_input_bpa('A', {
        frozenset({'h1'}): 0.1,
        frozenset({'h2'}): 0.5,
        frozenset({'h3'}): 0.3
    }, reliability=1)  # 添加可靠性参数
    
    dag.nodes['H'].set_input_bpa('N', {
        frozenset({'h1'}): 0.4,
        frozenset({'h2'}): 0.2,
        frozenset({'h3'}): 0.1
    }, reliability=1)  # 添加可靠性参数
    
    # 添加边
    dag.add_edge('E1', 'A', weight=1)
    dag.add_edge('A', 'H', weight=1)
    dag.add_edge('E2', 'A', weight=1)
    dag.add_edge('E4', 'InterMedia', weight=1)
    dag.add_edge('E5', 'InterMedia', weight=1)
    dag.add_edge('InterMedia', 'N', weight=1)
    dag.add_edge('E3', 'N', weight=1)
    dag.add_edge('N', 'H', weight=1)    

    print("带证据推理功能的有向无环图:")
    print(dag)
    
    # 计算节点可靠性
    print("\n节点可靠性计算:")
    for node_name in dag.get_nodes():
        [certification, reliability] = dag.calculate_node_reliability(node_name)
        print(f"节点 {node_name} 的可靠性:{reliability:.3f} ,确定性：{certification:.3f}")
    # 拓扑排序
    print("\n拓扑排序结果:")
    print(dag.topological_sort())

    # 计算最长路径
    longest_path, longest_weight = dag.longest_path('E1', 'H')
    print(f"\n从E1到H的最长路径: {longest_path}, 权重: {longest_weight}")
    
    # 传播证据
    hypothesis = {'a'}
    testnode = 'E1'
    results = dag.propagate_certification(testnode, hypothesis)
    
    print(f"\n 针对节点{testnode}中的假设{hypothesis} 的类概率函数={results:.3f}")
    
    # ========== 新增：测试 get_belief 和 get_spdf 方法 ==========
    print("\n" + "="*50)
    print("测试 get_belief 和 get_spdf 方法")
    print("="*50)
    
    # 测试节点 E1
    print(f"\n测试节点 E1:")
    node_e1 = dag.nodes['E1']
    
    # 测试单个假设
    hypothesis_a = {'a'}
    belief_a = node_e1.get_belief(hypothesis_a)
    plausibility_a = node_e1.get_plausibility(hypothesis_a)
    spdf_a = node_e1.get_spdf(hypothesis_a)  # 辨识框架维度为3
    
    print(f"假设 {hypothesis_a}:")
    print(f"  置信度 (Belief): {belief_a:.3f}")
    print(f"  似然度 (Plausibility): {plausibility_a:.3f}")
    print(f"  类概率函数 (SPDF): {spdf_a:.3f}")
    # 测试节点 H
    print(f"\n测试节点 H:")
    node_h = dag.nodes['H']
    
    # 测试单个假设
    hypothesis_h = {'h2'}
    belief_h = node_h.get_belief(hypothesis_h)
    plausibility_h = node_h.get_plausibility(hypothesis_h)
    spdf_h = node_h.get_spdf(hypothesis_h)  # 辨识框架维度为3
    
    print(f"假设 {hypothesis_h}:")
    print(f"  置信度 (Belief): {belief_h:.3f}")
    print(f"  似然度 (Plausibility): {plausibility_h:.3f}")
    print(f"  类概率函数 (SPDF): {spdf_h:.3f}")

    # 测试复合假设
    hypothesis_ab = {'a', 'b'}
    belief_ab = node_e1.get_belief(hypothesis_ab)
    plausibility_ab = node_e1.get_plausibility(hypothesis_ab)
    spdf_ab = node_e1.get_spdf(hypothesis_ab)
    
    print(f"假设 {hypothesis_ab}:")
    print(f"  置信度 (Belief): {belief_ab:.3f}")
    print(f"  似然度 (Plausibility): {plausibility_ab:.3f}")
    print(f"  类概率函数 (SPDF): {spdf_ab:.3f}")
    
    # 测试节点 E3
    print(f"\n测试节点 E3:")
    node_e3 = dag.nodes['E3']
    
    hypothesis_a_e3 = {'a'}
    belief_a_e3 = node_e3.get_belief(hypothesis_a_e3)
    plausibility_a_e3 = node_e3.get_plausibility(hypothesis_a_e3)
    spdf_a_e3 = node_e3.get_spdf(hypothesis_a_e3)  # 辨识框架维度为2
    
    print(f"假设 {hypothesis_a_e3}:")
    print(f"  置信度 (Belief): {belief_a_e3:.3f}")
    print(f"  似然度 (Plausibility): {plausibility_a_e3:.3f}")
    print(f"  类概率函数 (SPDF): {spdf_a_e3:.3f}")
    
    # 测试全集假设
    hypothesis_full_e3 = {'a', 'b'}
    belief_full_e3 = node_e3.get_belief()
    plausibility_full_e3 = node_e3.get_plausibility()
    spdf_full_e3 = node_e3.get_spdf()
    
    print(f"假设 {hypothesis_full_e3} (全集):")
    print(f"  置信度 (Belief): {belief_full_e3:.3f}")
    print(f"  似然度 (Plausibility): {plausibility_full_e3:.3f}")
    print(f"  类概率函数 (SPDF): {spdf_full_e3:.3f}")

    # 测试空集假设
    hypothesis_empty = set()
    try:
        belief_empty = node_e3.get_belief(hypothesis_empty)
        plausibility_empty = node_e3.get_plausibility(hypothesis_empty)
        spdf_empty = node_e3.get_spdf(hypothesis_empty)
        
        print(f"假设 {hypothesis_empty} (空集):")
        print(f"  置信度 (Belief): {belief_empty:.3f}")
        print(f"  似然度 (Plausibility): {plausibility_empty:.3f}")
        print(f"  类概率函数 (SPDF): {spdf_empty:.3f}")
    except ValueError as e:
        print(f"空集假设测试: {e}")
    
    # 测试多个节点的比较
    print(f"\n多个节点对假设 {{'a'}} 的比较:")
    test_nodes = ['E1', 'E3', 'E4', 'E5']
    for node_name in test_nodes:
        if node_name in dag.nodes:
            node = dag.nodes[node_name]
            try:
                belief = node.get_belief({'a'})
                plausibility = node.get_plausibility({'a'})
                spdf = node.get_spdf({'a'})
                
                print(f"节点 {node_name}: 置信度={belief:.3f}, 似然度={plausibility:.3f}, SPDF={spdf:.3f}")
            except Exception as e:
                print(f"节点 {node_name} 计算失败: {e}")
# ========== 新增：测试 EvidenceDAG 类的置信度、似然度和SPDF方法 ==========
    print("\n" + "="*60)
    print("测试 EvidenceDAG 类的置信度、似然度和SPDF方法")
    print("="*60)
    
    # 测试节点 E1 的折扣方法
    print(f"\n测试节点 E1 的折扣方法:")
    hypothesis_a = {'a'}
    
    # 原始BPA的置信度和似然度
    original_belief = dag.nodes['E1'].get_belief(hypothesis_a)
    original_plausibility = dag.nodes['E1'].get_plausibility(hypothesis_a)
    original_spdf = dag.nodes['E1'].get_spdf(hypothesis_a)
    
    print(f"原始BPA - 假设 {hypothesis_a}:")
    print(f"  置信度: {original_belief:.3f}")
    print(f"  似然度: {original_plausibility:.3f}")
    print(f"  SPDF: {original_spdf:.3f}")
    
    # 折扣后的置信度和似然度（使用节点自身可靠性）
    discounted_belief = dag.get_discounted_belief('E1', hypothesis_a)
    discounted_plausibility = dag.get_discounted_plausibility('E1', hypothesis_a)
    discounted_spdf = dag.get_discounted_spdf('E1', hypothesis_a)
    
    print(f"折扣后BPA - 假设 {hypothesis_a}:")
    print(f"  置信度: {discounted_belief:.3f}")
    print(f"  似然度: {discounted_plausibility:.3f}")
    print(f"  SPDF: {discounted_spdf:.3f}")
    
    # 使用不同可靠性值的折扣
    custom_reliability = 0.5
    custom_belief = dag.get_discounted_belief('E1', hypothesis_a, custom_reliability)
    custom_plausibility = dag.get_discounted_plausibility('E1', hypothesis_a, custom_reliability)
    custom_spdf = dag.get_discounted_spdf('E1', hypothesis_a, custom_reliability)
    
    print(f"自定义可靠性({custom_reliability})折扣 - 假设 {hypothesis_a}:")
    print(f"  置信度: {custom_belief:.3f}")
    print(f"  似然度: {custom_plausibility:.3f}")
    print(f"  SPDF: {custom_spdf:.3f}")
    
    # 测试传播后的方法
    print(f"\n测试传播后的方法:")
    propagated_belief = dag.get_propagated_belief('E1', hypothesis_a)
    propagated_plausibility = dag.get_propagated_plausibility('E1', hypothesis_a)
    propagated_spdf = dag.get_propagated_spdf('E1',hypothesis_a)
    
    print(f"传播后 - 假设 {hypothesis_a}:")
    print(f"  置信度: {propagated_belief:.3f}")
    print(f"  似然度: {propagated_plausibility:.3f}")
    print(f"  SPDF: {propagated_spdf:.3f}")
    
    # 比较不同节点的传播结果
    print(f"\n不同节点的传播结果比较 (假设 {{'a'}}):")
    test_nodes = ['E1', 'E2', 'E3', 'E4', 'E5']
    
    for node_name in test_nodes:
        if node_name in dag.nodes:
            node = dag.nodes[node_name]
            try:
                # 计算传播后的值
                p_belief = dag.get_propagated_belief(node_name, {'a'})
                p_plausibility = dag.get_propagated_plausibility(node_name, {'a'})
                p_spdf = dag.get_propagated_spdf(node_name, {'a'})
                
                # 计算节点可靠性
                [certification, reliability] = dag.calculate_node_reliability(node_name)
                
                print(f"节点 {node_name}: 可靠性={reliability:.3f}, 置信度={p_belief:.3f}, 似然度={p_plausibility:.3f}, SPDF={p_spdf:.3f}")
            except Exception as e:
                print(f"节点 {node_name} 计算失败: {e}")
    
    frame_set = frozenset()
    node = dag.nodes['A']
    for subset in node.bpa.keys():
        frame_set = frame_set.union(subset)

    p_belief = dag.get_propagated_belief('A')
    p_plausibility =dag.get_propagated_plausibility('A')
    p_spdf = dag.get_propagated_spdf('A')
    [certification, reliability] = dag.calculate_node_reliability('A')
    print(f"节点 'A': 可靠性={reliability:.3f}, 置信度={p_belief:.3f}, 似然度={p_plausibility:.3f}, SPDF={p_spdf:.3f}")
    p_belief = dag.get_propagated_belief('N')
    p_plausibility =dag.get_propagated_plausibility('N')
    p_spdf = dag.get_propagated_spdf('N')
    [certification, reliability] = dag.calculate_node_reliability('N')
    print(f"节点 'N': 可靠性={reliability:.3f}, 置信度={p_belief:.3f}, 似然度={p_plausibility:.3f}, SPDF={p_spdf:.3f}")
    p_belief = dag.get_propagated_belief('H')
    p_plausibility =dag.get_propagated_plausibility('H')
    p_spdf = dag.get_propagated_spdf('H')
    [certification, reliability] = dag.calculate_node_reliability('H')
    print(f"节点 'H': 可靠性={reliability:.3f}, 置信度={p_belief:.3f}, 似然度={p_plausibility:.3f}, SPDF={p_spdf:.3f}")

    p_belief = dag.get_propagated_belief('H')
    p_plausibility =dag.get_propagated_plausibility('H')
    p_spdf = dag.get_propagated_spdf('H')
    [certification, reliability] = dag.calculate_node_reliability('H')
    print(f"节点 'H': 可靠性={reliability:.3f},全集的 置信度={p_belief:.3f}, 似然度={p_plausibility:.3f}, SPDF={p_spdf:.3f}")

    # 创建证据DAG实例
    Evidence1 = destbasefunctionorigin.EvidenceNode('颜色分类1')
    Evidence1.set_frame({'红','黄','白'})
    Evidence1.set_bpa({
        frozenset({'红'}): 0.3,
        frozenset({'黄'}): 0.0,
        frozenset({'白'}): 0.1,
        frozenset({'红','黄'}): 0.2,
        frozenset({'红','白'}): 0.2,
        frozenset({'黄','白'}): 0,
        frozenset({'红','黄','白'}) : 0.2,
        frozenset({}): 0
    })

    belief = Evidence1.get_belief()
    print(f"颜色分类1:{belief:.3f}")

    belieftest= Evidence1.get_belief({'黄'})
    plausibility = Evidence1.get_plausibility({'黄'})
    print(f"节点的belief={belieftest:.3f}，似然度={plausibility:.3f}")

    Evidence2 = destbasefunctionorigin.EvidenceNode('颜色分类2')
    Evidence2.set_frame({'白','红','黄'})
    Evidence2.set_bpa({
        frozenset({'红'}): 0.3,
        frozenset({'黄'}): 0.2,
        frozenset({'白'}): 0.1,
        frozenset({'红','黄'}): 0.1,
        frozenset({'红','白'}): 0.1,
        frozenset({'黄','白'}): 0,
        frozenset({'红','黄','白'}) : 0.2,
        frozenset({}): 0
    })

    Evidence3 = destbasefunctionorigin.EvidenceNode('颜色分类3')
    Evidence3.set_frame({'鸡','鸭'})
    Evidence3.set_bpa({
        frozenset({'鸡'}): 0.4,
        frozenset({'鸭'}): 0.5,
        frozenset({'鸡','鸭'}): 0.1,
        frozenset({'∅'}): 0.0
        # frozenset({'Ω'}): 0.1        
    })
    belief = Evidence3.get_belief()
    print(f"颜色分类3:{belief:.3f}")

    Evidence4 = destbasefunctionorigin.EvidenceNode('颜色分类4')
    Evidence4.set_frame({'鸡','鸭'})
    Evidence4.set_bpa({
        frozenset({'∅'}): 0.0,
        frozenset({'鸡'}): 0.6,
        frozenset({'鸭'}): 0.2,
        frozenset({'鸡','鸭'}): 0.2
        # frozenset({'Ω'}): 0.1
    })
    
    belief = Evidence4.get_belief({'鸭','鸡'})
    print(f"belief is {belief:.3f}")
    plausibility = Evidence4.get_plausibility({'鸭','鸡'})
    print(f"plausibility is {plausibility:.3f}")
    spdf = Evidence4.get_spdf({'鸭','鸡'})
    print(f"spdf is {spdf:.3f}")

    belief = Evidence4.get_belief({'Ω'})
    print(f"belief of Ω is {belief:.3f}")
    plausibility = Evidence4.get_plausibility({'Ω'})
    print(f"plausibility of Ω is {plausibility:.3f}")
    spdf = Evidence4.get_spdf({'Ω'})
    print(f"spdf of Ω is {spdf:.3f}")

    belief = Evidence4.get_belief({'∅'})
    print(f"belief of ∅ is {belief:.3f}")
    plausibility = Evidence4.get_plausibility({'∅'})
    print(f"plausibility of ∅ is {plausibility:.3f}")
    spdf = Evidence4.get_spdf({'∅'})
    print(f"spdf of ∅ is {spdf:.3f}")

    combine_bpa = destbasefunctionorigin.EvidenceNode.merge_evidence_dempster(Evidence3.bpa,Evidence4.bpa,Evidence3.frame)
    print(combine_bpa)