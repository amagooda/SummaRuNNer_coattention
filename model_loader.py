from model import EncoderBiRNN
import torch

def init_model(params, vocab_size):
    summRunnerModel = EncoderBiRNN(params['batch_size'], vocab_size, params['embedding_size'], params['hidden_size'],
                                   max_num_sentence=params['max_num_sentences'], device=params['device'], use_bert=params['use_BERT'],
                                   num_bert_layers=len(params['BERT_layers']), bert_embedding_size=params['BERT_embedding_size'],
                                   use_coattention=params['use_coattention'], use_keywords=params['use_keywords'])
    if params['device'].type == 'cuda':
        summRunnerModel.cuda()
    return summRunnerModel

def reinit_embedding_layer(model, vocab_size, embedding_Size, max_number_sentences):
    model.reinit_embeddings(vocab_size, embedding_Size, max_number_sentences)
    return model


def save_model(model, optimizer, path, params):
    f = open(path, 'wb')
    state = {'params': params, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, f)
    f.close()


def load_model(optimizer, path, device):
    f = open(path, 'rb')
    state = torch.load(f, map_location=device.type)
    params = state['params']
    params['device'] = device
    model = init_model(params, params['vocab_size'])
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    model.load_state_dict(state['state_dict'], strict=False)
    optimizer.load_state_dict(state['optimizer'])
    f.close()

    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    print('Model Loaded....')
    return model, optimizer
