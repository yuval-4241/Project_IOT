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

    def add_symmetric_neighbor(self, neighbor):
        """הוספת שכן עם טבלת עלות סימטרית בין שני הסוכנים."""
        if neighbor.agent_id in self.neighbors:
            return  # לא להוסיף פעמיים

        self.neighbors.append(neighbor.agent_id)
        neighbor.neighbors.append(self.agent_id)

        table = {}
        for my_val in self.domain:
            for neighbor_val in neighbor.domain:
                if self.problem_type == 'general':
                    cost = random.randint(100, 200)
                elif self.problem_type == 'coloring':
                    cost = 0 if my_val != neighbor_val else 100
                table[(my_val, neighbor_val)] = cost
                table[(neighbor_val, my_val)] = cost  # סימטריה

        # שימור אותה טבלה אצל שני הסוכנים
        self.cost_tables[neighbor.agent_id] = table
        neighbor.cost_tables[self.agent_id] = table

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
        self.problem_type = problem_type
        self.seed = seed
        self.agents = []  # ניצור משתנה לשמירת הסוכנים
        if domain is not None:
            self.domain = domain
        elif self.problem_type == 'coloring':
            self.domain = ['red', 'green', 'blue','pink','yellow']  # דומיין קטן לצביעת גרף (3 צבעים)
        else:
            self.domain = ['a', 'b', 'c', 'd', 'e']  # דומיין רגיל

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
                    self.agents[i].add_symmetric_neighbor(self.agents[j])

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

    @staticmethod
    def draw_colored_graph(agents):
        import matplotlib.pyplot as plt
        import networkx as nx

        G = nx.Graph()

        for agent in agents:
            G.add_node(agent.agent_id)

        for agent in agents:
            for neighbor_id in agent.neighbors:
                if not G.has_edge(agent.agent_id, neighbor_id):
                    G.add_edge(agent.agent_id, neighbor_id)

        color_mapping = {
            'red': 'red',
            'blue': 'blue',
            'green': 'green',
            'pink': 'pink',
            'yellow': 'yellow'
        }

        node_colors = [color_mapping.get(agent.value.lower(), 'gray') for agent in agents]

        pos = nx.spring_layout(G, seed=42)
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=500, font_size=10)
        plt.title("גרף צבוע לפי ערכים שנבחרו ע\"י הסוכנים")
        plt.show()



###########################################
class DCOPEnvironment:
    def __init__(self, agents):
        self.agents = {agent.agent_id: agent for agent in agents}
        self.mailboxes = {agent.agent_id: [] for agent in agents}

    def receive_all_messages(self):
        for agent_id, agent in self.agents.items():
            # קבלת ההודעות מהתיבה של הסוכן
            agent.receive_messages(self.mailboxes[agent_id])
            print(f"📬 סוכן {agent_id} קיבל {len(self.mailboxes[agent_id])} הודעות מתוך {len(agent.neighbors)} שכנים")

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

    def run(self, p, algorithm='DSA', max_iterations=50):
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

            ##if changes == 0:
              ##  print(f"✅ התכנסות באיטרציה {iteration}")
                ##break
        min_iterations_before_checking_convergence = 5  # רוץ לפחות 5 איטרציות לפני בדיקת עצירה

        for iteration in range(max_iterations):
            # קבלת הודעות, ריצה, שליחת הודעות (כמו שיש אצלך)

            # בסוף איטרציה:
            if iteration >= min_iterations_before_checking_convergence and changes == 0:
                print(f"✅ התכנסות הושגה באיטרציה {iteration}")
                break




import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


# פונקציה: להריץ 30 בעיות ולחשב ממוצע על כל איטרציה
def average_costs_over_runs_shared_problems(ps=[0.2,0.7,1.0], k=0.25, problem_type='general', max_iterations=50, num_runs=50):
    results = {p: [] for p in ps}

    for run in range(num_runs):
        # יצירת בעיה עם seed קבוע
        creator = CreateEnvironment(problem_type=problem_type, seed=run)
        agents = creator.create_agents()
        creator.connect_agents(k=k)

        for p in ps:
            # שכפול של הסוכנים כדי לשמור על אותה התחלה
            copied_agents = deepcopy_agents(agents)  # אתה צריך לממש את זה!
            env = DCOPEnvironment(copied_agents)
            sim = Simulator(env)
            sim.run(p=p, algorithm='DSA', max_iterations=max_iterations)
            results[p].append(sim.costs_over_time)

    # החזרת ממוצע
    return {p: np.mean(np.array(results[p]), axis=0) for p in ps}

def deepcopy_agents(agents_list):
    from copy import deepcopy
    new_agents = []

    id_to_agent = {}

    for agent in agents_list:
        new_agent = Agent(
            agent_id=agent.agent_id,
            domain=deepcopy(agent.domain)
        )
        new_agent.value = agent.value
        new_agent.cost_tables = deepcopy(agent.cost_tables)
        id_to_agent[new_agent.agent_id] = new_agent
        new_agents.append(new_agent)

    # עדכון שכנים לפי מזהים
    for original_agent in agents_list:
        copied_agent = id_to_agent[original_agent.agent_id]
        copied_agent.neighbors = list(original_agent.neighbors)

    return new_agents



# פונקציה: מציירת את הגרף ל-k מסוים
def plot_dsa_for_k_fixed_problems(k, problem_type='general', save_as=None):
    ps = [ 0.2,0.7,1.0]
    max_iterations = 50
    results = average_costs_over_runs_shared_problems(ps=ps, k=k, problem_type=problem_type, max_iterations=max_iterations)

    plt.figure(figsize=(12, 6))
    for p in ps:
        plt.plot(range(1, max_iterations + 1), results[p], label=f'DSA-C (p={p})')
    plt.xlabel("Iteration")
    plt.ylabel("Average Global Cost")
    plt.title(f"DSA-C Comparison on k={k}" if problem_type != 'coloring' else "DSA-C on Graph Coloring (k=0.1)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if save_as:
        plt.savefig(save_as)
    else:
        plt.show()



# פונקציה ראשית: להריץ הכל
def run_all_dsa_graphs():
    plot_dsa_for_k_fixed_problems(k=0.25, problem_type='general')
    plot_dsa_for_k_fixed_problems(k=0.75, problem_type='general')
    plot_dsa_for_k_fixed_problems(k=0.1, problem_type='coloring')


# הפעלה
run_all_dsa_graphs()







