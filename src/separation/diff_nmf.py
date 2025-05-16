
import os
import subprocess
import numpy as np
import librosa
import soundfile as sf


mix_duration = 5.0
out_dir = 'results/demucs_generated'
model_name = 'htdemucs'  


def generate_mix():
 
    s1, sr = librosa.load(librosa.example('libri1'), duration=mix_duration)
    s2, _  = librosa.load(librosa.example('libri2'), duration=mix_duration)
    assert sr == sr, "fs be same for both sources"
    L = min(len(s1), len(s2))
    s1, s2 = s1[:L], s2[:L]
    sources = np.vstack([s1, s2])
    A = np.array([[1.0, 0.5], [0.5, 1.0]])
    X = A @ sources                # (2, L)
    mix = X.mean(axis=0)           # (L,)
    return mix, sr


def separate_with_cli(mix_path, out_dir, model_name='htdemucs'):

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        'demucs',           
        '-n', model_name,   
        '--two-stems', 'vocals',
        '-o', out_dir,
        mix_path
    ]
    subprocess.run(cmd, check=True)

    result_dir = os.path.join(
        out_dir,
        model_name,
        os.path.splitext(os.path.basename(mix_path))[0]
    )
    vocals = os.path.join(result_dir, 'vocals.wav')
    accompaniment = os.path.join(result_dir, 'accompaniment.wav')
    if os.path.exists(vocals) and os.path.exists(accompaniment):
        sf.write(os.path.join(out_dir, 'source1.wav'), *sf.read(vocals))
        sf.write(os.path.join(out_dir, 'source2.wav'), *sf.read(accompaniment))
        print(f"CLI separation done, sources at {out_dir}")
    else:
        print(f"Expected outputs not found in {result_dir}")


if __name__ == '__main__':
    mix, sr = generate_mix()
    mix_path = os.path.join(out_dir, 'mix.wav')
    sf.write(mix_path, mix, sr)
    print("Generated mix of length", mix.shape[0], "at sr", sr)
    separate_with_cli(mix_path, out_dir, model_name)
