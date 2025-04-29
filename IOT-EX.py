import random
import networkx as nx
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, agent_id, domain, problem_type='general'):
        self.agent_id = agent_id
        self.domain = domain[:]
        self.problem_type = problem_type
        self.value = random.choice(self.domain)
        self.neighbors = []  # רק רשימת IDs של שכנים
        self.cost_tables = {}  # עלויות מול שכנים
        self.inbox = []  # תיבת דואר - הודעות נכנסות באיטרציה הנוכחית

    def add_neighbor(self, neighbor):
        """הוספת שכן והגדרת עלויות בהתאם לסוג הבעיה."""
        self.neighbors.append(neighbor.agent_id)
        table = {}
        for my_val in self.domain:
            for neighbor_val in neighbor.domain:
                if self.problem_type == 'general':
                    cost = random.randint(100, 200)
                elif self.problem_type == 'coloring':
                    cost = 0 if my_val != neighbor_val else 100
                table[(my_val, neighbor_val)] = cost
        self.cost_tables[neighbor.agent_id] = table


    def receive_messages(self, messages):
        self.inbox = messages  # כל ההודעות שקיבלתי באיטרציה

    def compute_cost(self, my_value, received_messages):
        """חישוב עלות נוכחית או עבור ערך אלטרנטיבי."""
        total_cost = 0
        for msg in received_messages:
            neighbor_id = msg['sender']
            neighbor_value = msg['value']
            cost_table = self.cost_tables.get(neighbor_id)
            if cost_table is not None:
                cost = cost_table.get((my_value, neighbor_value), 0)
                total_cost += cost
            else:
                # אם אין טבלה בכלל לשכן – מתעלמים
                continue
        return total_cost

    def run_dsa(self, p):
        outgoing_messages = []

        current_cost = self.compute_cost(self.value, self.inbox)
        best_value = self.value
        best_cost = current_cost

        for value in self.domain:
            if value == self.value:
                continue
            cost = self.compute_cost(value, self.inbox)
            if cost < best_cost:
                best_value = value
                best_cost = cost

        changed = False
        old_value = self.value
        if best_value != self.value:
            if random.random() < p:
                self.value = best_value
                changed = True

        for neighbor_id in self.neighbors:
            outgoing_messages.append({
                'sender': self.agent_id,
                'receiver': neighbor_id,
                'value': self.value
            })

        return outgoing_messages, changed, old_value, self.value


####################################3

import random
import networkx as nx
import matplotlib.pyplot as plt

class CreateEnvironment:
    def __init__(self, num_agents=30, domain=None, problem_type='general', seed=123):
        self.num_agents = num_agents
        self.domain = domain if domain is not None else ['a', 'b', 'c', 'd', 'e']
        self.problem_type = problem_type
        self.seed = seed
        self.agents = []  # ניצור משתנה לשמירת הסוכנים

    def create_agents(self):
        random.seed(self.seed)
        self.agents = [
            Agent(agent_id=i, domain=self.domain, problem_type=self.problem_type)
            for i in range(1, self.num_agents + 1)
        ]
        print("\n--- סוכנים וערכים התחלתיים ---")
        for agent in self.agents:
            print(f"סוכן {agent.agent_id}: ערך התחלתי = {agent.value}")
        return self.agents

    def connect_agents(self, k, seed=42):
        random.seed(seed)
        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                if random.random() < k:
                    self.agents[i].add_neighbor(self.agents[j])
                    self.agents[j].add_neighbor(self.agents[i])
        print("\n--- רשימת שכנויות ---")
        for agent in self.agents:
            print(f"סוכן {agent.agent_id}: שכנים = {agent.neighbors}")
        return self.agents

    def draw_graph(self):
        """מצייר את הגרף של הסוכנים והחיבורים ביניהם."""
        G = nx.Graph()

        # הוספת קודקודים
        for agent in self.agents:
            G.add_node(agent.agent_id)

        # הוספת קשתות
        for agent in self.agents:
            for neighbor_id in agent.neighbors:
                if not G.has_edge(agent.agent_id, neighbor_id):
                    G.add_edge(agent.agent_id, neighbor_id)

        # ציור
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
        plt.title("רשת סוכנים - חיבורים בין סוכנים")
        plt.show()



###########################################
class DCOPEnvironment:
    def __init__(self, agents):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.mailboxes = {agent.agent_id: [] for agent in agents}

    def receive_all_messages(self):
        """נותן לכל סוכן את ההודעות שהגיעו אליו"""
        for agent_id, agent in self.agents.items():
            agent.receive_messages(self.mailboxes[agent_id])

    def run_all_agents(self, p, algorithm='DSA'):
        """מריץ את כל הסוכנים ומנהל שליחת הודעות"""
        new_mailboxes = {agent_id: [] for agent_id in self.agents}
        changes = 0

        for agent_id, agent in self.agents.items():
            if algorithm == 'DSA':
                outgoing_messages, changed = agent.run_dsa(p)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            for msg in outgoing_messages:
                new_mailboxes[msg['receiver']].append(msg)

            if changed:
                changes += 1

        self.mailboxes = new_mailboxes
        return changes

    def get_global_cost(self):
        """חישוב עלות כוללת של הרשת"""
        total_cost = 0
        visited_pairs = set()

        for agent_id, agent in self.agents.items():
            for neighbor_id in agent.neighbors:
                if (agent_id, neighbor_id) not in visited_pairs and (neighbor_id, agent_id) not in visited_pairs:
                    my_value = agent.value
                    neighbor_value = self.agents[neighbor_id].value
                    cost = agent.cost_tables[neighbor_id].get((my_value, neighbor_value), 0)
                    total_cost += cost
                    visited_pairs.add((agent_id, neighbor_id))

        return total_cost

    def print_cost_tables(self):
        print("\n--- טבלאות עלויות לכל סוכן מול שכניו ---")
        for agent_id, agent in self.agents.items():
            print(f"\nסוכן {agent_id}:")
            for neighbor_id, table in agent.cost_tables.items():
                print(f"  מול שכן {neighbor_id}:")
                for (my_val, neighbor_val), cost in sorted(table.items()):
                    print(f"    ({my_val}, {neighbor_val}) → {cost}")


###############################################################



class Simulator:
    def __init__(self, environment):
        self.environment = environment
        self.costs_over_time = []

    def run(self, p=0.7, algorithm='DSA', max_iterations=50):
        print("\n--- מתחילים סימולציה ---")

        for iteration in range(max_iterations):
            print(f"\n>>> איטרציה {iteration} <<<")

            # שלב קבלת הודעות
            self.environment.receive_all_messages()

            # שלב ריצה
            changes = 0
            detailed_changes = []

            new_mailboxes = {agent_id: [] for agent_id in self.environment.agents}

            for agent_id, agent in self.environment.agents.items():
                if algorithm == 'DSA':
                    outgoing_messages, changed, old_value, new_value = agent.run_dsa(p)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")

                for msg in outgoing_messages:
                    new_mailboxes[msg['receiver']].append(msg)

                if changed:
                    changes += 1
                    detailed_changes.append((agent_id, old_value, new_value))

            self.environment.mailboxes = new_mailboxes

            # שמירת עלות כוללת
            global_cost = self.environment.get_global_cost()
            self.costs_over_time.append(global_cost)

            # הדפסה מפורטת
            if detailed_changes:
                print(f"🌀 {changes} סוכנים שינו ערך:")
                for agent_id, old, new in detailed_changes:
                    print(f"  סוכן {agent_id}: {old} → {new}")
            else:
                print("✅ אף סוכן לא שינה ערך - התכנסות!")

            print(f"💰 עלות כוללת ברשת: {global_cost}")

            if changes == 0:
                print(f"✅ התכנסות באיטרציה {iteration}")
                break

    def plot_costs(self):
        """מצייר את התקדמות העלות לאורך האיטרציות."""
        plt.plot(self.costs_over_time, marker='o')
        plt.title('Global Cost Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Global Cost')
        plt.grid(True)
        plt.show()


#########################
# 1. יצירת סוכנים
creator = CreateEnvironment(problem_type='general', seed=123)
agents = creator.create_agents()
agents = creator.connect_agents(agents, k=0.25)

# 2. יצירת Environment
env = DCOPEnvironment(agents)
creator.draw_graph()
# 3. יצירת סימולטור והרצה
simulator = Simulator(env)
simulator.run(p=0.7, algorithm='DSA')

# 4. ציור גרף ירידת עלות
simulator.plot_costs()


