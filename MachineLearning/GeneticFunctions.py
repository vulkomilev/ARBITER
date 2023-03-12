import random

class GeneticAlg(object):
    def __int__(self):
        self.genes_lib = []
        self.gene_to_func_map = {
            '0000':{'name':'add','arg_len':2},
            '0001':{'name': 'concat','arg_len':2},
            '0002':{'name': 'contains','arg_len':2},
            '0003':{'name': 'truediv','arg_len':2},
            '0004':{'name': 'floordiv','arg_len':2},
            '0005':{'name': 'and_','arg_len':2},
            '0006':{'name': 'xor','arg_len':2},
            '0007':{'name': 'invert','arg_len':2},
            '0008':{'name': 'or_','arg_len':2},
            '0009':{'name': 'pow','arg_len':2},
            '0010':{'name': 'is_','arg_len':2},
            '0011':{'name': 'is_not','arg_len':2},
            '0012':{'name': 'setitem','arg_len':2},
            '0013':{'name': 'delitem','arg_len':2},
            '0014':{'name': 'getitem','arg_len':2},
            '0015':{'name': 'lshift','arg_len':2},
            '0016':{'name': 'mod','arg_len':2},
            '0017':{'name': 'mul','arg_len':2},
            '0018':{'name': 'matmul','arg_len':2},
            '0019':{'name': 'neg','arg_len':2},
            '0020':{'name': 'not_','arg_len':2},
            '0021':{'name': 'pos','arg_len':2},
            '0022':{'name': 'rshift','arg_len':2},
            '0023':{'name': 'setitem','arg_len':2},
            '0024':{'name': 'delitem','arg_len':2},
            '0025':{'name': 'getitem','arg_len':2},
            '0026':{'name': 'mod','arg_len':2},
            '0027':{'name': 'sub','arg_len':2},
            '0028':{'name': 'truth','arg_len':2},
            '0029':{'name': 'lt','arg_len':2},
            '0030':{'name': 'le','arg_len':2},
            '0031':{'name': 'eq','arg_len':2},
            '0032':{'name': 'ne','arg_len':2},
            '0033':{'name': 'ge','arg_len':2},
            '0034':{'name': 'gt','arg_len':2},
            '0035':{'name': 'abs','arg_len':2},
            '0036':{'name': 'aiter','arg_len':2},
            '0037':{'name': 'all','arg_len':2},
            '0038':{'name': 'any','arg_len':2},
            '0039':{'name': 'anext','arg_len':2},
            '0040':{'name': 'ascii','arg_len':2},
            '0041':{'name': 'bin','arg_len':2},
            '0042':{'name': 'bool','arg_len':2},
            '0043':{'name': 'breakpoint','arg_len':2},
            '0044':{'name': 'bytearray','arg_len':2},
            '0045':{'name': 'bytes','arg_len':2},
            '0046':{'name': 'callable','arg_len':2},
            '0047':{'name': 'chr','arg_len':2},
            '0048':{'name': 'classmethod','arg_len':2},
            '0049':{'name': 'compile','arg_len':2},
            '0050':{'name': 'complex','arg_len':2},
            '0051':{'name': 'delattr','arg_len':2},
            '0052':{'name': 'dict','arg_len':2},
            '0053':{'name': 'dir','arg_len':2},
            '0054':{'name': 'divmod','arg_len':2},
            '0055':{'name': 'enumerate','arg_len':2},
            '0056':{'name': 'eval','arg_len':2},
            '0057':{'name': 'exec','arg_len':2},
            '0058':{'name': 'filter','arg_len':2},
            '0059':{'name': 'float','arg_len':2},
            '0060':{'name': 'format','arg_len':2},
            '0061':{'name': 'frozenset','arg_len':2},
            '0062':{'name': 'getattr','arg_len':2},
            '0063':{'name': 'globals','arg_len':2},
            '0064':{'name': 'hasattr','arg_len':2},
            '0065':{'name': 'hash','arg_len':2},
            '0066':{'name': 'help','arg_len':2},
            '0067':{'name': 'hex','arg_len':2},
            '0068':{'name': 'id','arg_len':2},
            '0069':{'name': 'input','arg_len':2},
            '0070':{'name': 'int','arg_len':2},
            '0071':{'name': 'isinstance','arg_len':2},
            '0072':{'name': 'issubclass','arg_len':2},
            '0073':{'name': 'iter','arg_len':2},
            '0074':{'name': 'len','arg_len':2},
            '0075':{'name': 'list','arg_len':2},
            '0076':{'name': 'locals','arg_len':2},
            '0077':{'name': 'map','arg_len':2},
            '0078':{'name': 'max','arg_len':2},
            '0079':{'name': 'memoryview','arg_len':2},
            '0080':{'name': 'min','arg_len':2},
            '0081':{'name': 'next','arg_len':2},
            '0082':{'name': 'object','arg_len':2},
            '0083':{'name': 'oct','arg_len':2},
            '0084':{'name': 'open','arg_len':2},
            '0085':{'name': 'ord','arg_len':2},
            '0086':{'name': 'pow','arg_len':2},
            '0087':{'name': 'print','arg_len':2},
            '0088':{'name': 'property','arg_len':2},
            '0089':{'name': 'range','arg_len':2},
            '0090':{'name': 'repr','arg_len':2},
            '0091':{'name': 'reversed','arg_len':2},
            '0092':{'name': 'round','arg_len':2},
            '0093':{'name': 'set','arg_len':2},
            '0094':{'name': 'setattr','arg_len':2},
            '0095':{'name': 'slice','arg_len':2},
            '0096':{'name': 'sorted','arg_len':2},
            '0097':{'name': 'staticmethod','arg_len':2},
            '0098':{'name': 'str','arg_len':2},
            '0099':{'name': 'sum','arg_len':2},
            '0100':{'name': 'super','arg_len':2},
            '0101':{'name': 'tuple','arg_len':2},
            '0102':{'name': 'type','arg_len':2},
            '0103':{'name': 'vars','arg_len':2},
            '0104':{'name': 'zip','arg_len':2},
            '0105':{'name': '__import__','arg_len':2},
            #-----------
            '1000':{'letter':'\n'},
            '1001': {'letter': '\t'},
            '1002': {'letter': '='},
            '1003': {'letter': '+='},
            '1004': {'letter': '-='},
            '1005': {'letter': '/='},
            '1006': {'letter': '*='},
            '1007': {'letter': '-'},
            '1008': {'letter': '+'},
            '1009': {'letter': '/'},
            '1010': {'letter': '*'},
            '1011': {'letter': '**'},
            '1012': {'letter': '%'}
        }

    def random_name(self,name_len=10):
        local_name = ''
        for i in range(name_len):
            local_name +=  random.choice('qwertyuiopasdfghjklzxcvbnm1234567890')
        return local_name

    def generate_function_from_gene(self,gene):

        str_func = '''               
        '''
        genes = [gene[i:i+4] for i in range(0, len(gene), 4)]
        for local_gene in genes:
            if local_gene[0] in '1':
               str_func+= self.gene_to_func_map[local_gene]['letter']
            elif local_gene[0] in '0':
               str_func+= self.gene_to_func_map[local_gene]['name']
            elif local_gene[0] in '2':
                if local_gene not in  list(self.gene_to_func_map.keys()):
                    self.gene_to_func_map[local_gene] = {'name':self.random_name()}

               str_func+= self.gene_to_func_map[local_gene]['name']