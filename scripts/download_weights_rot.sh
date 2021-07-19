# download networks weights and seed data (for Pose Server)

cd src/pose_lib

if [ ! -d train.configs ]; then
    mkdir train.configs
fi

cd train.configs

if [ ! -f 106_index.npy ]; then
    megadl 'https://mega.nz/#!6BhilC5S!hZ_nf1jSyBkp2ZGi7y9LjegPciknXm2PuosZl6HYF4I'
fi

if [ ! -f keypoints_sim.npy ]; then
    megadl 'https://mega.nz/#!aJxmjSrZ!0Z3F0LWshPPXZl9RFzXcCPQ35nmFodpHyA1YfbfXUdg'
fi

if [ ! -f Model_PAF.npy ]; then
    megadl 'https://mega.nz/#!KV4CTATI!3P9Ra_jyfRAxpyulwFdonzpW0HzxwdfZBRydblVc3s0'
fi

if [ ! -f param_whitening.pkl ]; then
    megadl 'https://mega.nz/#!HZYDHQBA!5uHRos0ZH8CR03jUpCirU_JzuBRpy1MWpeo4t4Btzq4'
fi

if [ ! -f pncc_code.npy ]; then
    megadl 'https://mega.nz/#!TEggACqR!hdu4Wal_j5CXY-wCJAkBb7ijw2mk8cvcpVIUN6-pDVA'
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
