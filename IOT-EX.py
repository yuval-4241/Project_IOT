import random
import networkx as nx
import matplotlib.pyplot as plt

# הגדרת הסתברות לחיבור בין סוכנים
k = 0.25

# מחלקת Agent
class Agent:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.domain = ['a', 'b', 'c', 'd', 'e']
        self.value = random.choice(self.domain)
        self.mailbox = []
        self.neighbors = []
        self.cost_tables = {}

    def add_neighbor(self, neighbor):
        self.neighbors.append(neighbor)
        table = {}
        for my_val in self.domain:
            for neighbor_val in neighbor.domain:
                table[(my_val, neighbor_val)] = random.randint(100, 200)
        self.cost_tables[neighbor.agent_id] = table

    def send_value_to_neighbors(self):
        for neighbor in self.neighbors:
            neighbor.mailbox.append((self.agent_id, self.value))

    def receive_messages(self):
        received = self.mailbox
        self.mailbox = []
        return received

    def __repr__(self):
        return f"Agent({self.agent_id}): Value={self.value}"

# יצירת 30 סוכנים (החל מ-1)
agents = [Agent(agent_id=i) for i in range(1, 31)]

# חיבור בין כל זוג סוכנים לפי הסתברות k
for i in range(len(agents)):
    for j in range(i + 1, len(agents)):
        if random.random() < k:
            agents[i].add_neighbor(agents[j])
            agents[j].add_neighbor(agents[i])

# הדפסת טבלאות עלויות
for agent in agents:
    print(f"\nAgent {agent.agent_id} Cost Tables:")
    for neighbor_id, cost_table in agent.cost_tables.items():
        print(f"  Neighbor {neighbor_id}:")
        for (my_val, neighbor_val), cost in cost_table.items():
            print(f"    ({my_val}, {neighbor_val}) -> {cost}")

# יצירת גרף לציור
G = nx.Graph()

# הוספת קודקודים
for agent in agents:
    G.add_node(agent.agent_id)

# הוספת קשתות בין שכנים
for agent in agents:
    for neighbor in agent.neighbors:
        if not G.has_edge(agent.agent_id, neighbor.agent_id):
            G.add_edge(agent.agent_id, neighbor.agent_id)

# ציור הגרף
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
plt.title(f"רשת סוכנים עם k={k}")
plt.show()
