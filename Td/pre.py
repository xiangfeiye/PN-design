import torch
from trainer_2 import *
from data2 import Vocab
from utils import *
from tqdm import trange
gastype = {'N2': 'dq',
            'air': 'air',
            'Ar': 'ar'}
splitor = smiles_split()

def single_pre(smiles, dsrate=10, loss=5, gas=gastype['N2'], mn = None):
    new_smiles, split_smiles, smiles_len = get_new_smiles(smiles, splitor=splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)
    tmp =vocab.text2tensor(new_smiles)
    source = torch.zeros(seq_length).int()
    source[:len(tmp)] = torch.tensor(tmp).int()
    source = source.to(device)
    pre = model(source.unsqueeze(0))
    pred = pre * std + mean
    return pred[0][0].item()

def batch_pre(smiles_list, dsrate=10, loss=5, gas=gastype['N2'], mn = None):
    sources = torch.zeros(len(smiles_list), seq_length).int()
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        new_smiles, split_smiles, smiles_len = get_new_smiles(smiles, splitor=splitor, dsrate=dsrate, loss=loss, gas=gas, mn=mn)
        tmp =vocab.text2tensor(new_smiles)
        sources[i][:len(tmp)] = torch.tensor(tmp).int()
        
    sources = sources.to(device)
    with torch.no_grad():
        pre = model(sources)
    pred = pre * std + mean
    return pred
    
    
def smi2smi(smiles, chiral=True, H=False):
    m = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(m, isomericSmiles=chiral, allHsExplicit=H)
    return smi

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    model = torch.load('Td_LSTMModelPre.pt').to(device)
    model.eval()
    vocab = torch.load('vocab.pt')
    seq_length = model.max_len
    [mean, std] = model.scaler
    mean = mean.to(device)
    std = std.to(device)

    df = pd.read_csv('NO2_O2_results.csv')
    batch_size = 128
    smiles_ = list(df['smiles'])

    results = []
    for j in trange(len(df)//batch_size + 1):
        smiles_list = smiles_[j*batch_size:min(len(df), (j+1)*batch_size)]
        smiles_list = [smi2smi(smiles) for smiles in smiles_list]
        pred = batch_pre(smiles_list).view(-1)
        
        results += list(pred.view(-1).detach().cpu().numpy())

    df['td5'] = results
    df.to_csv('NO2_O2_results_td5.csv', index=False)
