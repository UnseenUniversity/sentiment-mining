
import json
from dependency_parser.jsonrpc import ServerProxy, JsonRpc20, TransportTcpIp
from nltk.tree import Tree
from pprint import pprint

class StanfordNLP:
    def __init__(self):
        self.server = ServerProxy(JsonRpc20(),
                                  TransportTcpIp(addr=("127.0.0.1", 8080)))

    def parse(self, text):
        return json.loads(self.server.parse(text))

    def get_tree(self, text):

        stree = self.parse(text)
        tree = Tree.parse(stree['sentences'][0]['parsetree'])


def _test():
    nlp = StanfordNLP()
    result = nlp.parse("Keneddt hasn't been killed. ")#She doesn't look very beautiful and charming but she is gorgeous.")
    #result = nlp.parse("I strongly believe it was very good however I wasn't impressed.")
    pprint(result)

    # from nltk.tree import Tree
    # tree = Tree.parse(result['sentences'][0]['parsetree'])
    # pprint(tree)

if __name__ == "__main__":
    _test()