import json
import re
import codecs
import numpy as np
import random
import padding_utils

def read_text_file(text_file):
    lines = []
    with open(text_file, "rt") as f:
        for line in f:
            line = line.decode('utf-8')
            lines.append(line.strip())
    return lines

def read_nary_file(inpath, options):
    all_words = []
    all_lemmas = []
    all_poses = []
    all_in_neigh = []
    all_in_label = []
    all_out_neigh = []  # [batch, node, neigh]
    all_out_label = []  # [batch, node, neigh]
    all_entity_indices = []  # [batch, 3, entity_size]
    all_y = []
    if options.class_num == 2:
        relation_set = {'resistance or non-response': 0, 'sensitivity': 0, 'response': 0, 'resistance': 0, 'None': 1, }
    elif options.class_num == 5:
        relation_set = {'resistance or non-response': 0, 'sensitivity': 1, 'response': 2, 'resistance': 3, 'None': 4, }
    else:
        assert False, 'Illegal class num'
    max_words = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_entity_size = 0

    all_inst = []
    all_entity_tuples = []
    with codecs.open(inpath, 'rU', 'utf-8') as f:
        for inst in json.load(f):
            words = []
            lemmas = []
            poses = []
            if options.only_single_sent and len(inst['sentences']) > 1:
                continue
            all_inst.append(inst)
            for sentence in inst['sentences']:
                for node in sentence['nodes']:
                    words.append(node['label'])
                    lemmas.append(node['lemma'])
                    poses.append(node['postag'])
            max_words = max(max_words, len(words))
            all_words.append(words)
            all_lemmas.append(lemmas)
            all_poses.append(poses)
            in_neigh = [[i, ] for i, _ in enumerate(words)]
            in_label = [['self', ] for i, _ in enumerate(words)]
            out_neigh = [[i, ] for i, _ in enumerate(words)]
            out_label = [['self', ] for i, _ in enumerate(words)]
            for sentence in inst['sentences']:
                for node in sentence['nodes']:
                    i = node['index']
                    for arc in node['arcs']:
                        j = arc['toIndex']
                        l = arc['label']
                        l = l.split('::')[0]
                        l = l.split('_')[0]
                        l = l.split('(')[0]
                        if j == -1 or l == '':
                            continue
                        in_neigh[j].append(i)
                        in_label[j].append(l)
                        out_neigh[i].append(j)
                        out_label[i].append(l)
            for _i in in_neigh:
                max_in_neigh = max(max_in_neigh, len(_i))
            for _o in out_neigh:
                max_out_neigh = max(max_out_neigh, len(_o))
            all_in_neigh.append(in_neigh)
            all_in_label.append(in_label)
            all_out_neigh.append(out_neigh)
            all_out_label.append(out_label)
            entity_indices = []
            entity_words = []
            for entity in inst['entities']:
                entity_indices.append(entity['indices'])
                entity_words.append(entity['mention'])
                max_entity_size = max(max_entity_size, len(entity['indices']))
            assert len(entity_indices) == options.entity_num
            all_entity_indices.append(entity_indices)
            all_entity_tuples.append([entity_words, relation_set[inst['relationLabel'].strip()]])
            all_y.append(relation_set[inst['relationLabel'].strip()])
    all_lex = all_lemmas if options.word_format == 'lemma' else all_words
    # def tupleCompare(e1,e2):
    #     if e1[0]!=e2[0]:
    #         return 1 if e1[0]>e2[0] else -1
    #     elif e1[1]!=e2[1]:
    #         return 1 if e1[1] > e2[1] else -1
    #     elif e1[2]!=e2[2]:
    #         return 1 if e1[2] > e2[2] else -1
    #     else:
    #         return 0
    # all_entity_tuples.sort(cmp=tupleCompare)

    # with open(inpath+"_tuples", "w") as f:
    #     json.dump(all_entity_tuples, f)
    return zip(all_lex, all_poses, all_in_neigh, all_in_label, all_out_neigh, all_out_label, all_entity_indices, all_y), \
           max_words, max_in_neigh, max_out_neigh, max_entity_size, all_entity_tuples, all_inst


def read_nary_from_fof_init(fofpath, options):
    all_paths = read_text_file(fofpath)
    all_instances = []
    all_entity_tuples = []
    max_words = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_entity_size = 0
    all_init_inst = []
    for cur_path in all_paths:
        print(cur_path)
        cur_instances, cur_words, cur_in_neigh, cur_out_neigh, cur_entity_size, cur_entity_tuples, cur_init_inst = read_nary_file(
            cur_path, options)
        all_instances.extend(cur_instances)
        all_init_inst.extend(cur_init_inst)
        all_entity_tuples.extend(cur_entity_tuples)
        max_words = max(max_words, cur_words)
        max_in_neigh = max(max_in_neigh, cur_in_neigh)
        max_out_neigh = max(max_out_neigh, cur_out_neigh)
        max_entity_size = max(max_entity_size, cur_entity_size)
    return all_instances, max_words, max_in_neigh, max_out_neigh, max_entity_size, all_init_inst


def read_nary_from_fof(fofpath, options):
    all_paths = read_text_file(fofpath)
    all_instances = []
    all_entity_tuples = []
    max_words = 0
    max_in_neigh = 0
    max_out_neigh = 0
    max_entity_size = 0
    all_init_inst = []
    for cur_path in all_paths:
        print(cur_path)
        cur_instances, cur_words, cur_in_neigh, cur_out_neigh, cur_entity_size, cur_entity_tuples, cur_init_inst = read_nary_file(
            cur_path, options)
        all_instances.extend(cur_instances)
        all_init_inst.extend(cur_init_inst)
        all_entity_tuples.extend(cur_entity_tuples)
        max_words = max(max_words, cur_words)
        max_in_neigh = max(max_in_neigh, cur_in_neigh)
        max_out_neigh = max(max_out_neigh, cur_out_neigh)
        max_entity_size = max(max_entity_size, cur_entity_size)

    # dump all the tuples
    # all_tuples=[]
    # for t in all_entity_tuples:
    #     all_tuples.append(t[0])
    # with codecs.open("data/all_tuples", 'w', 'UTF-8') as f:
    #     json.dump(all_tuples, f)
    # print("total {} instances".format(len(all_tuples)))



    # classify instance according to relation subtype and
    # from collections import defaultdict
    # classified_inst = [defaultdict(list) for _ in range(4)]
    # classified_index = [defaultdict(list) for _ in range(4)]
    # for i, t in enumerate(all_entity_tuples):
    #     key = "_".join([t[0][0], t[0][1], t[0][2]])
    #     classified_index[t[1]][key].append(i)
    #     classified_inst[t[1]][key].append(all_init_inst[i])
    # print("total {} classes".format(len(classified_inst)))
    #
    # # use 1/5 of the tuples in every class as the test set, rest as pool set;
    # # use 1/5 of tuples in every class in pool set as train set, rest as pool set;
    # test_inst = []
    # test_ind = []
    # pool_inst = []
    # pool_ind = []
    # train_inst = []
    # train_ind = []
    # test_tuples = [[] for _ in range(4)]
    # pool_tuples = [[] for _ in range(4)]
    # train_tuples = [[] for _ in range(4)]
    # for i in range(4):
    #     # split the dataset according to number of instances
    #     # curr_test_tuples = {}
    #     # for k,v in classified_index[i].items():
    #     #     if len(v)>=4 and len(v)<100:
    #     #         curr_test_tuples[k]=v
    #     #     if len(curr_test_tuples)>=2: break
    #     # curr_pool_tuples = {}
    #     # for k,v in classified_index[i].items():
    #     #     if k not in curr_test_tuples:
    #     #         curr_pool_tuples[k]=v
    #     # curr_train_tuples = {}
    #     # for k,v in curr_pool_tuples.items():
    #     #     if len(v)>=4 and len(v)<100:
    #     #         curr_train_tuples[k]=v
    #     #     if len(curr_train_tuples)>=2: break
    #
    #     # split the dataset tuples randomly
    #     all_class_tuples = list(classified_index[i])
    #     split_test = int(0.2 * len(all_class_tuples))
    #     curr_test_tuples = all_class_tuples[:split_test]
    #     curr_pool_tuples = all_class_tuples[split_test:]
    #     split_train = int(0.2 * len(curr_pool_tuples))
    #     curr_train_tuples = curr_pool_tuples[:split_train]
    #
    #     # get the instances
    #     for k in curr_test_tuples:
    #         test_tuples[i].append(k)
    #         v=classified_index[i][k]
    #         test_ind.extend(v)
    #         for ind in v:
    #             test_inst.append(all_init_inst[ind])
    #     for k in curr_pool_tuples:
    #         pool_tuples[i].append(k)
    #         v = classified_index[i][k]
    #         pool_ind.extend(v)
    #         for ind in v:
    #             pool_inst.append(all_init_inst[ind])
    #     for k in curr_train_tuples:
    #         train_tuples[i].append(k)
    #         v = classified_index[i][k]
    #         train_ind.extend(v)
    #         for ind in v:
    #             train_inst.append(all_init_inst[ind])
    #
    # # dump to the files
    # with codecs.open("data/test_tuple", 'w', 'UTF-8') as f:
    #     json.dump(test_tuples, f)
    # with codecs.open("data/pool_tuple", 'w', 'UTF-8') as f:
    #     json.dump(pool_tuples, f)
    # with codecs.open("data/train_tuple", 'w', 'UTF-8') as f:
    #     json.dump(train_tuples, f)
    # with codecs.open("data/test_inst", 'w', 'UTF-8') as f:
    #     json.dump(test_inst, f)
    # with codecs.open("data/test_ind", 'w', 'UTF-8') as f:
    #     json.dump(test_ind, f)
    # with codecs.open("data/pool_inst", 'w', 'UTF-8') as f:
    #     json.dump(pool_inst, f)
    # with codecs.open("data/pool_ind", 'w', 'UTF-8') as f:
    #     json.dump(pool_ind, f)
    # with codecs.open("data/train_ind", 'w', 'UTF-8') as f:
    #     json.dump(train_ind, f)
    # with codecs.open("data/train_inst", 'w', 'UTF-8') as f:
    #     json.dump(train_inst, f)
    # # with codecs.open("data/class_train_"+str(i),'w','UTF-8') as f:
    # #     json.dump(classified_inst[i],f)
    # # with codecs.open("data/class_train_" + str(i)+"_index", 'w', 'UTF-8') as f:
    # #     json.dump(classified_index[i],f)
    #
    # # find the number of tuples
    # num_tuples=0
    # tuple_dict=defaultdict(list)
    # for i, t in enumerate(all_entity_tuples):
    #     key="_".join([t[0][0],t[0][1],t[0][2]])
    #     if key in tuple_dict:
    #         tuple_dict[key]+=1
    #         continue
    #     tuple_dict[key]=1
    #     num_tuples+=1
    # with codecs.open("data/all_tuples_positive",'w','utf-8') as f:
    #     json.dump(tuple_dict,f,indent=4,sort_keys=True)
    #
    # # find the number of non-crossing-subtype tuples
    # tuple_class={}
    # cross_subtype_tuples=set()
    # for t in all_entity_tuples:
    #     key="_".join([t[0][0],t[0][1],t[0][2]])
    #     if key in cross_subtype_tuples: continue
    #     if key in tuple_class:
    #         if tuple_class[key]!=t[1]:
    #             cross_subtype_tuples.add(key)
    #             continue
    #     tuple_class[key]=t[1]
    # for k in cross_subtype_tuples:
    #     tuple_class.pop(k)
    # with codecs.open("data/all_tuples_without_multiple_subtypes",'w','utf-8') as f:
    #     json.dump(tuple_class,f,sort_keys=True,indent=4)
    # print('num of tuples: {}'.format(len(tuple_dict)))
    # print('num of tuples without multiple subtypes: {}'.format(len(tuple_class)))

    # get the number of instances without crossing subtypes
    # num_tuples_unique=0
    # for k in tuple_class.keys():
    #     num_tuples_unique+=tuple_dict[k]
    # print('num of instances without multiple subtypes: {}'.format(num_tuples_unique))

    #     key_list=list(tuple_dict.keys())
    #     key_list.sort()
    #     for k in key_list:
    #         f.write(k+" "+str(tuple_dict[k])+"\n")
    # print("num of tuples: ",num_tuples)

    return all_instances, max_words, max_in_neigh, max_out_neigh, max_entity_size


def collect_vocabs(all_instances):
    all_words = set()
    all_chars = set()
    all_edgelabels = set()
    for (lex, poses, in_neigh, in_label, out_neigh, out_label, entity_indices, y) in all_instances:
        all_words.update(lex)
        for l in lex:
            if l.isspace() == False: all_chars.update(l)
        for edges in in_label:
            all_edgelabels.update(edges)
        for edges in out_label:
            all_edgelabels.update(edges)
    return (all_words, all_chars, all_edgelabels)

class G2SDataStream(object):
    def __init__(self, all_instances, word_vocab=None, char_vocab=None, edgelabel_vocab=None, options=None,
                 isShuffle=False, isLoop=False, isSort=True, batch_size=-1):
        self.options = options
        if batch_size == -1: batch_size = options.batch_size
        # index tokens and filter the dataset
        instances = []
        for (lex, poses, in_neigh, in_label, out_neigh, out_label, entity_indices, y) in all_instances:
            if options.max_node_num != -1 and len(lex) > options.max_node_num:
                continue  # remove very long passages
            in_neigh = [x[:options.max_in_neigh_num] for x in in_neigh]
            in_label = [x[:options.max_in_neigh_num] for x in in_label]
            out_neigh = [x[:options.max_out_neigh_num] for x in out_neigh]
            out_label = [x[:options.max_out_neigh_num] for x in out_label]

            lex_idx = word_vocab.to_index_sequence_for_list(lex)
            lex_chars_idx = None
            if options.with_char:
                lex_chars_idx = char_vocab.to_character_matrix_for_list(lex,
                                                                        max_char_per_word=options.max_char_per_word)
            in_label_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in in_label]
            out_label_idx = [edgelabel_vocab.to_index_sequence_for_list(edges) for edges in out_label]
            instances.append(
                (lex_idx, lex_chars_idx, in_neigh, in_label_idx, out_neigh, out_label_idx, entity_indices, y))

        all_instances = instances
        instances = None

        # sort instances based on length
        if isSort:
            all_instances = sorted(all_instances, key=lambda inst: len(inst[0]))
        if isShuffle:
            random.shuffle(all_instances)
            random.shuffle(all_instances)
        self.num_instances = len(all_instances)

        # distribute questions into different buckets
        batch_spans = padding_utils.make_batches(self.num_instances, batch_size)
        self.batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = []
            for i in xrange(batch_start, batch_end):
                cur_instances.append(all_instances[i])
            cur_batch = G2SBatch(cur_instances, options, word_vocab=word_vocab)
            self.batches.append(cur_batch)

        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.isShuffle = isShuffle
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.isLoop = isLoop
        self.cur_pointer = 0

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
            if self.isShuffle: np.random.shuffle(self.index_array)
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        if self.isShuffle: np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[i]


class G2SBatch(object):
    def __init__(self, instances, options, word_vocab=None):
        self.options = options

        self.instances = instances  # list of tuples
        self.batch_size = len(instances)
        self.vocab = word_vocab

        # node num
        self.node_num = []  # [batch_size]
        for (lex_idx, lex_chars_idx, in_neigh, in_label_idx, out_neigh, out_label_idx, entity_indices, y) in instances:
            self.node_num.append(len(lex_idx))
        self.node_num = np.array(self.node_num, dtype=np.int32)

        # node char num
        if options.with_char:
            self.nodes_chars_num = [[len(lex_chars_idx) for lex_chars_idx in instance[1]] for instance in instances]
            self.nodes_chars_num = padding_utils.pad_2d_vals_no_size(self.nodes_chars_num)

        # neigh mask
        self.in_neigh_mask = []  # [batch_size, node_num, neigh_num]
        self.out_neigh_mask = []
        self.entity_indices_mask = []
        for instance in instances:
            ins = []
            for in_neighs in instance[2]:
                ins.append([1 for _ in in_neighs])
            self.in_neigh_mask.append(ins)
            outs = []
            for out_neighs in instance[4]:
                outs.append([1 for _ in out_neighs])
            self.out_neigh_mask.append(outs)
            idxs = []
            for entity_indices in instance[6]:
                idxs.append([1 for _ in entity_indices])
            self.entity_indices_mask.append(idxs)
        self.in_neigh_mask = padding_utils.pad_3d_vals_no_size(self.in_neigh_mask)
        self.out_neigh_mask = padding_utils.pad_3d_vals_no_size(self.out_neigh_mask)
        self.entity_indices_mask = padding_utils.pad_3d_vals_no_size(self.entity_indices_mask)

        # the actual contents
        self.nodes = [x[0] for x in instances]
        if options.with_char:
            self.nodes_chars = [x[1] for x in instances]  # [batch_size, sent_len, char_num]
        self.in_neigh_indices = [x[2] for x in instances]
        self.in_neigh_edges = [x[3] for x in instances]
        self.out_neigh_indices = [x[4] for x in instances]
        self.out_neigh_edges = [x[5] for x in instances]
        self.entity_indices = [x[6] for x in instances]
        self.y = [x[7] for x in instances]

        # making ndarray
        self.nodes = padding_utils.pad_2d_vals_no_size(self.nodes)
        if options.with_char:
            self.nodes_chars = padding_utils.pad_3d_vals_no_size(self.nodes_chars)
        self.in_neigh_indices = padding_utils.pad_3d_vals_no_size(self.in_neigh_indices)
        self.in_neigh_edges = padding_utils.pad_3d_vals_no_size(self.in_neigh_edges)
        self.out_neigh_indices = padding_utils.pad_3d_vals_no_size(self.out_neigh_indices)
        self.out_neigh_edges = padding_utils.pad_3d_vals_no_size(self.out_neigh_edges)
        self.entity_indices = padding_utils.pad_3d_vals_no_size(self.entity_indices)
        self.y = np.asarray(self.y, dtype='int32')

        assert self.in_neigh_mask.shape == self.in_neigh_indices.shape
        assert self.in_neigh_mask.shape == self.in_neigh_edges.shape
        assert self.out_neigh_mask.shape == self.out_neigh_indices.shape
        assert self.out_neigh_mask.shape == self.out_neigh_edges.shape
        assert self.entity_indices_mask.shape == self.entity_indices.shape

        assert self.entity_indices.shape[1] == options.entity_num
        assert self.entity_indices_mask.shape[1] == options.entity_num

    def get_amrside_anonyids(self, anony_ids):
        assert self.batch_size == 1  # only for beam search
        if self.options.__dict__.has_key("enc_word_vec_path"):
            assert self.options.enc_word_vec_path == self.options.dec_word_vec_path  # only when enc_vocab == dec_vocab
        self.amr_anony_ids = set(self.instances[0][0]) & anony_ids  # sent1 of inst_0


if __name__ == "__main__":
    all_instances, max_node_num, max_in_neigh_num, max_out_neigh_num, max_entity_size = read_nary_from_fof(
        './data/data_list', 'lemma')
    print sum(len(x[0]) for x in all_instances) / len(all_instances)
    print(max_in_neigh_num)
    print(max_out_neigh_num)
    print(max_node_num)
    print(max_entity_size)
    print('DONE!')

