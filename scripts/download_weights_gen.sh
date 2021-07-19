# download networks weights and seed data (for StyleGAN2 Server)

cd src/stylegan_lib

if [ ! -d models ]; then
    mkdir models
fi

cd models

if [ ! -f stgan2_model.pt ]; then
    megadl 'https://mega.nz/#!aExSxQAa!K-X4sgm-suacm9LZlnMctQj5vgZHp7f_Ry4KM7GbhfM'
fi

if [ ! -f initial_seed.npy ]; then
    megadl 'https://mega.nz/#!3ZwSlLTC!vlbC_I_kw2l5bLGwPuXRM5_nerbJ9WPHeDmhU04l9qQ'
fi

cd ../text_processing

if [ ! -d checkpoints ]; then
    mkdir checkpoints
fi

cd checkpoints

if [ ! -f distilbert-base-uncased.pth ]; then
    megadl 'https://mega.nz/#!CchDlSJD!Oe3QmtiTCRaqqEIqZSXxlcoqycbXKka2mAjxUq3yJGI'
fi

cd ../../../..
