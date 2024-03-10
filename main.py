from scripts.generate_song_data_beam import generate_song_db
from absl import app
def main(argv):
    generate_song_db(argv)
if __name__ == '__main__':
    app.run(main)