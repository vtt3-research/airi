

def decode_captions(captions, idx_to_word):
    if captions.ndim == 1:
        len_cap = captions.shape[0]
        num_cap = 1
    else:
        num_cap, len_cap = captions.shape

    decoded = []
    for i in range(num_cap):
        words = []
        for t in range(len_cap):
            if captions.ndim == 1:
                word = idx_to_word[captions[t]]
            else:
                word = idx_to_word[captions[i, t]]
            if word == '<END>':
                words.append('.')
                break
            if word != '<NULL>':
                words.append(word)
        decoded.append(' '.join(words))
    return decoded


def normalized_encoding(captions, word_to_idx):
    start_number = word_to_idx['<START>']
    end_number = word_to_idx['<END>']
    null_number = word_to_idx['<NULL>']

    if captions.ndim == 1:
        len_cap = captions.shape[0]
        num_cap = 1
    else:
        num_cap, len_cap = captions.shape

    norm_encoded = []
    for i in range(num_cap):
        words = []
        words.append(start_number)
        end_flag = False
        for t in range(len_cap):
            if captions.ndim == 1:
                word = captions[t]
            else:
                word = captions[i, t]
            if end_flag:
                words.append(null_number)
            elif word == end_number:
                end_flag = True
                words.append(end_number)
            else:
                words.append(word)
        norm_encoded.append(words)
    return norm_encoded


def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if arr[i] == 0:
            break
    return out.strip()


def array_to_str_for_score(arr):
    start_number = 1
    end_number = 2
    null_number = 0
    out = ''
    for i in range(len(arr)):
        if arr[i] == start_number:
            continue
        elif arr[i] == null_number:
            break
        elif arr[i] == end_number:
            out += str(arr[i]) + ' '
            break
        else:
            out += str(arr[i]) + ' '
    return out.strip()
