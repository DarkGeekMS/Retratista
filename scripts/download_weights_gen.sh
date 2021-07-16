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

cd ../text_processing/pybert/output/checkpoints/bert

if [ ! -f pytorch_model.bin ]; then
    megadl 'https://mega.nz/#!iQ4iBBYA!2k7bWbeZLixzt3c_ry2sHhHBahol7nL8RRBeoGksu9c'
fi

cd ../../../../../../..
