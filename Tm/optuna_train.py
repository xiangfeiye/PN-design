from trainer_2 import *
import optuna


def objective(trial):
    data_path = '../data/'
    filename = 'en_isomer_21m_2_300'
    byfile = True
    max_len = 300
    x_name = 'smiles'
    other_name = []
    y_name = ['mp']
    vocab_name = 'vocab.pt'
    data_name = 'en_isomer_21m_2_300'
    standard = True

    seed = 42
    train_ratio = [0.8, 0.1, 0.1]
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    loader_shuffle = True
    drop_last = True

    n_layers = trial.suggest_int("n_layers", 1, 8, step=1)
    input_size = 2 ** trial.suggest_int("input_size", 3, 8, step=1)
    decoder_size = [trial.suggest_int("decoder_size1", 64, 512, step=64),
                    trial.suggest_int("decoder_size2", 8, 64, step=8)]

    pre_dropout = trial.suggest_float("pre_dropout", 0, 0.4, step=0.1)
    hidden_size = trial.suggest_int("hidden_size", 64, 512, step=64)
    dropout = trial.suggest_float("dropout", 0, 0.4, step=0.1)
    bidirectional = trial.suggest_categorical("bidirectional", [True, False])
    proj_size = 0

    learningrate = [trial.suggest_float("lr", 1e-5, 1e-2, log=True),
                    trial.suggest_float("step", 4, 32, step=4),
                    trial.suggest_float("step_ratio", 0.2, 0.8, step=0.1)]

    use_gpu = True
    log_freq = 10
    epochs = 32
    output_path = None

    device = torch.device('cuda:0' if use_gpu else 'cpu')

    print('load data')
    mydataset = CommonDataset(path=data_path,
                              filename=filename,
                              byfile=byfile,
                              max_len=max_len,
                              x_name=x_name,
                              other_name=other_name,
                              y_name=y_name,
                              device=None,
                              vocab_name=vocab_name,
                              dataname=data_name,
                              standard=standard)

    traindataloader, testdataloader, validdataloader = dataset2dataloader(mydataset, train_ratio, batch_size,
                                                                          loader_shuffle, drop_last, seed)

    print('build predict model')
    # pre_model = TransformerPre(vocab_size=mydataset.get_vocab_len(), hidden=hidden_size, n_layers=n_layers,
    #                            attn_heads=attn_heads,
    #                            padding_idx=mydataset.vocab.stoi['pad'], max_len=mydataset.max_len, dropout=dropout,
    #                            decoder_size=decoder_size, pre_dropout=pre_dropout, target_num=len(mydataset.y_name),
    #                            device=device)
    pre_model = LSTMModelPre(ntoken=mydataset.get_vocab_len(), max_len=mydataset.max_len, input_size=input_size,
                             hidden_size=hidden_size, num_layers=n_layers, dropout=dropout,
                             bidirectional=bidirectional, proj_size=proj_size, mlp_dropout=pre_dropout,
                             decoder_size=decoder_size, target_num=len(mydataset.y_name),
                             device=device)

    pre_model.scaler = mydataset.scaler if mydataset.scaler is not None else None

    print("Creating Trainer")
    trainer = LSTMModelPreTrainer(pre_model, mydataset.vocab, train_dataloader=traindataloader,
                                  test_dataloader=testdataloader, valid_dataloader=validdataloader, lr=learningrate,
                                  use_gpu=use_gpu, log_freq=log_freq, ignore_index=mydataset.vocab.stoi['pad'])

    print("Training Start")
    print('epoch, train/test, avg_loss, avg_mae, avg_r2, avg_rmse')

    stop_flag = 0
    for epoch in range(epochs):
        loss1 = trainer.train(epoch)
        loss2 = trainer.test(epoch)
        trainer.lr_scheduler.step()

        trial.report(loss2, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if loss1 < loss2:
            stop_flag += 1
        if stop_flag >= 3:
            break

        # trainer.save(epoch, output_path)
    print('valid start')
    loss = trainer.valid(epochs)
    print(loss)

    return loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    study = optuna.create_study(pruner=optuna.pruners.MedianPruner(
        n_startup_trials=6, n_warmup_steps=6, interval_steps=20), )
    study.optimize(objective, n_trials=64)
    best_params = study.best_params
    print(best_params)
    torch.save(study, 'result.pt')
