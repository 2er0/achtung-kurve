from sklearn import tree
import pickle
import graphviz

agent = 33

with open("agent/"+str(agent)+"/agent.pkl", "rb") as fp:
    other = pickle.load(fp)

tree_data = tree.export_graphviz(other.clf, out_file=None)
graph = graphviz.Source(tree_data)
graph.view("agent33")
