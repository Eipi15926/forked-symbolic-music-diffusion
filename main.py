from scripts import generate_song_data_beam
from absl import app
def main(argv):
    generate_song_data_beam.main(argv)
if __name__ == '__main__':
    app.run(main)