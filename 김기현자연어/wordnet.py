import torch
from nltk.corpus import wordnet as wn

def hypernyms(word):
    current_node = wn.synsets(word)[0]
    yield current_node

    while True:
        try:
            current_node = current_node.hypernyms()[0]
            yield current_node
        except IndexError:
            break
hypernyms(wn.synsets('policeman')[0])

for ss in wn.synsets('bass'):
    print(ss, ss.definition())
print("="*20)

# 단어중의성해소(WSD)

# 레스크 알고리즘 수행 - 가장 단단한 사전 기반 중의성 해소 방법
# 사전의 단어 및 의미에 관한 설명에 크게 의존.
# 설명이 부실하거나 주어진 문장에 큰 특징이 없을 경우 단어 중의성 해소 능력이 크게 떨어짐. 사전이 존재하지 않는 언어에 대해서는 레스크 알고리즘 자체 수행 불가능
def lesk(sentence, word):
    from nltk.wsd import lesk

    best_synset = lesk(sentence.split(), word)
    print(best_synset, best_synset.definition())
sentence = 'I went fishing last weekend and I got a bass and cooked it'
word = 'bass'
lesk(sentence, word)

sentence2 = 'I love the music from te speaker which has strong beat and bass'
word2 = 'bass'
lesk(sentence2,word2)
print("="*20)

sentence3 = 'I think the bass is more important tha guitar'
word3 = 'bass'
lesk(sentence3, word3)

# 선택 선호도
# 각 단어는 문장 내 주변의 단어들에 따라 그 의미가 정해짐 -> 이를 더 수치화
# 선택 선호도 강도는 술어가 표제어로 특정 클래스를 얼마나 선ㄴ택적으로 선호하는지에 대한 수치

# 유사도 기반의 선택 선호도 예제
import konlpy
from konlpy.tag import Mecab, Kkma

mecab=konlpy.tag.Mecab(dicpath=r"C:/mecab/mecab-ko-dic")
def count_seen_headwords(lines, predicate='VV',headword='NNG'): # VV = 술어(동사), NNG = 표제어(명사)
    tagger = Kkma()
    seen_dict={}

    for line in lines:
        pos_result = tagger.pos(line)

        word_h = None
        word_p = None
        for word, pos in pos_result:
            if pos == predicate or pos[:3] == predicate + '+':
                word_p = word
                break
            if pos == headword:
                word_h = word
            if word_h is not None and word_p is not None:
                seen_dict[word_p] = [word_h] + ([] if seen_dict.get(word_p) is None else seen_dict[word_p])

        return seen_dict

# 선택관련도 점수를 구하는 함수
def get_selectional_association(predicate, headword, lines, dataframe, metric):
    # 단어 사이의 유사도를 구하기위해, 이전에 구성한 특징 벡터들을 담은 pandas dataframe을 받음.
    # metric으로 주어진 함수를 통해 유사도를 계산
    v1=torch.FloatTensor(dataframe.loc[headword].values)
    seens = count_seen_headwords[predicate]

    total=0
    for seen in seens:
        try:
            v2 = torch.FloatTensor(dataframe.loc[seen].values)
            total += metric(v1,v2)
        except:
            pass

    return total
# 주어진 술어에 대해 올바른 headword를 고르는 wsd 함수
def wsd(predicate, headwords):
    selectional_associatations = []
    for h in headwords:
        selectional_associatations += [get_selectional_association(predicate, h, lines, co, cosine_similarity)]
    print(selectional_associatations)
