# download networks weights and seed data (for Pose Server)

cd src/pose_lib

if [ ! -d train.configs ]; then
    mkdir train.configs
fi

cd train.configs

if [ ! -f keypoints_sim.npy ]; then
    megadl 'https://mega.nz/#!aJxmjSrZ!0Z3F0LWshPPXZl9RFzXcCPQ35nmFodpHyA1YfbfXUdg'
fi

if [ ! -f param_all_norm_val.pkl ]; then
    megadl 'https://mega.nz/#!3BBF0CzR!6isKcTLdUwjon5qJBZaVZfcd9KAXmwdRnfMNAUEcXQo'
fi

if [ ! -f param_all_norm.pkl ]; then
    megadl 'https://mega.nz/#!aF5EEQzT!fICPDvwd14Mqp4958xxgbaGo8cl8m6fc28LDb2wqNmM'
fi

if [ ! -f param_whitening.pkl ]; then
    megadl 'https://mega.nz/#!HZYDHQBA!5uHRos0ZH8CR03jUpCirU_JzuBRpy1MWpeo4t4Btzq4'
fi

if [ ! -f train_aug_120x120.list.train ]; then
    megadl 'https://mega.nz/#!WYpkUYCQ!0Eg5mQmr7B3VGuU9v69_sadcMFRUxRHhnFvD6utaVeI'
fi

if [ ! -f train_aug_120x120.list.val ]; then
    megadl 'https://mega.nz/#!SVxWDagb!gpl8npJUly0ozQvKPzTvlhvSggJZYIYlVDYZpTe-1N0'
fi

if [ ! -f u_exp.npy ]; then
    megadl 'https://mega.nz/#!7ApmTSrL!Wl2VVgxE3VuVr-hsZGg5oOE08NJQpJXU9_RJPhcKsBY'
fi

if [ ! -f u_shp.npy ]; then
    megadl 'https://mega.nz/#!DVJlRIYC!n7kNgkjIn2hG-T6M2KXOp8WyJrz4YgvBPwyn_-NQOWQ'
fi

if [ ! -f w_exp_sim.npy ]; then
    megadl 'https://mega.nz/#!nIRDQYCY!mbATmmVOZElAVa2QDcY2NtOovd_L_yZCFX7b0VkvnQg'
fi

if [ ! -f w_shp_sim.npy ]; then
    megadl 'https://mega.nz/#!2BIXGSrK!KDAtaXBSgOLm0NI1ATf1uurBFIAWFHbJ-NoqfePlt9g'
fi

cd ..

if [ ! -f phase1_pdc.pth.tar ]; then
    megadl 'https://mega.nz/#!3I4RmY7C!t72Q_-ocBLyqcuY_oFb2sT_YpeoC9R2tKqOuMKlQ9uk'
fi

if [ ! -d checkpoints/rs_model ]; then
    mkdir -p checkpoints/rs_model
fi

cd checkpoints/rs_model

if [ ! -f latest_net_G.pth ]; then
    megadl 'https://mega.nz/#!3MZUBJ5a!oE9T_Z01o41Cxdj81td_qHvHVb52DOFCBGzqhGuSobI'
fi

cd ../../../..
