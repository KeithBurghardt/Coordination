import os
import socket

if __name__ == '__main__':
    for seed in [
        # 0,
        # 1, 2, 3, 4, 5, 6, 7, 8, 9,
        2022
                 ]:
        data = ['vast', 'pstance', 'covid'][2]
        inference = 0
        chunk_size = -1
        # seed = 0
        # topic = ['bernie', 'biden', 'trump', 'all'][-1]
        # topic = ['bernie,biden', 'bernie,trump', 'biden,bernie', 'biden,trump', 'trump,bernie', 'trump,biden'][5]
        topic = ['face_masks', 'fauci', 'stay_at_home_orders', 'school_closures', 'all'][-1]
        batch_size = 16
        epochs = 100
        patience = 10
        lr = 2e-5
        l2_reg = 5e-5
        max_grad = 0
        model = ['bert-base', 'bertweet', 'covid-twitter-bert', 'bertweet-covid', 'roberta', 'bertweetfr', 'bert-small',
                 'xlm-t'][-1]
        wiki_model = \
        ['', 'bert-base', 'bertweet', 'covid-twitter-bert', 'bertweet-covid', 'roberta', 'bertweetfr', 'bert-small',
         'merge'][1]
        n_layers_freeze = 0
        n_layers_freeze_wiki = 11
        gpu = '0,1,2,3'

        if wiki_model == model:
            n_layers_freeze_wiki = n_layers_freeze
        if not wiki_model or wiki_model == 'merge':
            n_layers_freeze_wiki = 0

        os.makedirs('results', exist_ok=True)
        file_name = f'results/lr={lr}-bs={batch_size}_seed={seed}_infer={inference}.txt'

        if max_grad > 0:
            file_name = file_name[:-4] + f'-max_grad={max_grad}.txt'
        if model != 'bert-base':
            file_name = file_name[:-4] + f'-{model}.txt'
        if n_layers_freeze > 0:
            file_name = file_name[:-4] + f'-n_layers_fz={n_layers_freeze}.txt'
        if wiki_model:
            file_name = file_name[:-4] + f'-wiki={wiki_model}.txt'
        if n_layers_freeze_wiki > 0:
            file_name = file_name[:-4] + f'-n_layers_fz_wiki={n_layers_freeze_wiki}.txt'

        n_gpus = len(gpu.split(','))
        file_name = file_name[:-4] + f'-n_gpus={n_gpus}.txt'

        command = f"python3 -u train.py " \
                  f"--data={data} " \
                  f"--topic={topic} " \
                  f"--model={model} " \
                  f"--wiki_model={wiki_model} " \
                  f"--n_layers_freeze={n_layers_freeze} " \
                  f"--n_layers_freeze_wiki={n_layers_freeze_wiki} " \
                  f"--batch_size={batch_size} " \
                  f"--epochs={epochs} " \
                  f"--patience={patience} " \
                  f"--lr={lr} " \
                  f"--l2_reg={l2_reg} " \
                  f"--max_grad={max_grad} " \
                  f"--gpu={gpu} " \
                  f"--seed={seed} " \
                  f"--inference={inference} " \
            f" > {file_name}"

        print(command)
        os.system(command)

