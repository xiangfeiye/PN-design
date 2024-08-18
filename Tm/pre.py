from data import *
from rdkit import Chem
from data import Vocab
from tqdm import trange

def GetdataFromSmiles(smiles_list, vocab, max_len):
    datas = torch.zeros((len(smiles_list), max_len))
    x_lens = torch.zeros(len(smiles_list))
    for i in range(len(smiles_list)):
        smiles = smiles_list[i]
        data = torch.tensor(vocab.smi2tensor(smiles))
        x_len = len(data)
        x_lens[i] = x_len
        pad1 = nn.ConstantPad1d((0, max_len - x_len), 0)
        datas[i, :] = pad1(data)
    return datas.int(), x_lens.int()

def smilereplace(smile):
    replace_dict = {
                    'Si': 'V',
                    "Cl": 'U',
                    "Br": 'K',
                    "@@": 'D',
                    }
    for key in replace_dict.keys():
        smile = smile.replace(key, replace_dict[key])

    return smile

def smi2smi(smiles):
    smi = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smi

def pre_single(smiles):
    smi_list = [smiles]
    datas, x_lens = GetdataFromSmiles(smi_list, vocab, max_len=pre_model.max_len)
    datas = datas.to(device)
    x_lens = x_lens.to(device)

    pre = pre_model([datas, x_lens])
    pred = pre.cpu().detach().numpy().reshape(-1) * std + mean
    
    return pre, pred

def pre_batch(smiles):
    datas, x_lens = GetdataFromSmiles(smiles, vocab, max_len=pre_model.max_len)
    datas = datas.to(device)
    x_lens = x_lens.to(device)

    pre = pre_model([datas, x_lens])
    pred = pre.cpu().detach().numpy().reshape(-1) * std + mean
    
    return pre, pred

def smi2smi(smiles, chiral=True, H=False):
    m = Chem.MolFromSmiles(smiles)
    smi = Chem.MolToSmiles(m, isomericSmiles=chiral, allHsExplicit=H)
    return smi

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    device = torch.device('cuda')
    pre_model = torch.load('Tm_LSTMModelPre.pt').to(device)
    pre_model.eval()
    vocab = torch.load('vocab.pt')
    mean = pre_model.scaler[0].item()
    std = pre_model.scaler[1].item()

    df = pd.read_csv('../NO2_O2_results.csv')
    mps = []
    for i in trange(len(df)//512+1):
        smiles = list(df['smiles'])[i*512:min((i+1)*512, len(df))]
        _, pred = pre_batch(smiles)
        mps += list(pred)

    df['mp_1211'] = mps
    df.to_csv('NO2_O2_results_mp.csv', index=False)
