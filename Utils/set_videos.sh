python Utils/getFrames.py --dataFolder data_video --videoFolder videos/inigo_reves --className inigo  --framerRate 11
python Utils/rotate_images_left.py --folder data_video
python Utils/rotate_images_left.py --folder data_video
# las que estan al reves se ponen como todas (no rectas porque despues e vovleran a girar)
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/inigo --className inigo  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/marcos --className marcos  --framerRate 13
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/jorge --className jorge  --framerRate 13
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/agus --className agus  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/anaCris --className anaCris  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/natalia --className natalia  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/lara --className lara  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/maite --className maite  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/anaCambra --className anaCambra  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/brandom --className brandom  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/hardware-desconocido --className hardware-desconocido  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/dario --className dario  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/ronie --className ronie  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/rubenGran --className rubenGran  --framerRate 11
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/Steven --className Steven  --framerRate 9
python Utils/getFrames.py --dataFolder data_video --videoFolder videos/pilar --className pilar  --framerRate 11
python Utils/gen_train_test_files.py --dataFolder data_video
python Utils/squareImages.py --dataFolder data_video
python Utils/resizeImages.py --dataFolder data_video --width 224 --height 224
python Utils/rotate_images_left.py --folder data_video
