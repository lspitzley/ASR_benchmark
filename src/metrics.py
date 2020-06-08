import numpy
import string
import logging
from num2words import num2words

def normalize_text(text, lower_case=False, remove_punctuation=False, write_numbers_in_letters=True):
    '''
    Perform text normalization
    '''
    if lower_case: text = text.lower()

    # https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string-in-python
    table = str.maketrans({key: " " for key in string.punctuation})
    if remove_punctuation: text = text.translate(table)

    if write_numbers_in_letters:
        
        # adapted from: https://www.geeksforgeeks.org/python-extract-numbers-from-string/
        text_list = text.split()
        num_indices = [i for i in range(len(text_list)) if text_list[i].isnumeric()] 

        for i in num_indices:
            text_list[i] = num2words(text_list[i])

        # reconstruct text
        text = ' '.join(text_list)
    return text

def wer2(r, h):
    '''
    This function was originally written by Martin Thoma
    https://martin-thoma.com/word-error-rate-calculation/

    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list
    h : list

    Returns
    -------
    int

    Examples
    --------
    >>> wer("who is there".split(), "is there".split())
    1
    >>> wer("who is there".split(), "".split())
    3
    >>> wer("".split(), "who is there".split())
    3
    '''
    # Initialization
    d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
    d = d.reshape((len(r)+1, len(h)+1))
    for i in range(len(r)+1):
        for j in range(len(h)+1):
            if i == 0:
                d[0][j] = j
            elif j == 0:
                d[i][0] = i

    # Computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                substitution = d[i-1][j-1] + 1
                insertion    = d[i][j-1] + 1
                deletion     = d[i-1][j] + 1
                d[i][j] = min(substitution, insertion, deletion)

    return d[len(r)][len(h)]


def wer(ref, hyp ,debug=False):
    '''
    This function was originally written by SpacePineapple
    http://progfruits.blogspot.com/2014/02/word-error-rate-wer-and-word.html
    '''
    DEL_PENALTY = 1
    SUB_PENALTY = 1
    INS_PENALTY = 1
    #r = ref.split()
    #h = hyp.split()
    r= ref
    h = hyp
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]

    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3

    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL

    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS

    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1]
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1

                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL

    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")

    lines.append("OP\tREF\tHYP")
    #return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    #return {'WER':wer_result, 'Cor':numCor, 'Sub':numSub, 'Ins':numIns, 'Del':numDel}
    return {'changes': numSub + numDel + numIns, 'corrects':numCor, 
            'substitutions':numSub, 'insertions':numIns, 'deletions':numDel,
            'wer_result': wer_result}, lines


if __name__ == "__main__":
    import doctest
    #doctest.testmod()
    print(wer("who is there".split(), "is there a cat".split()))
    print(wer("who is there".split(), "who is cat".split()))
    print(wer("blue had range of okay".split(), "blue hydrangea bouquet".split()))
    print(wer("blue hydrangea bouquet".split(), "blue had range of okay".split()))
