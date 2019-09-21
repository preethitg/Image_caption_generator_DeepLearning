from name_to_job import *
from job_to_salary import *
from keras.models import load_model
import tensorflow.contrib.eager as tfe
from tensorflow.python.eager.context import context, EAGER_MODE, GRAPH_MODE
def switch_to(mode):
    ctx = context()._eager_context
    ctx.mode = mode
    ctx.is_eager = mode == EAGER_MODE

def main():
    last_name = input('Enter your family name: ')
    print('Family name:  ' + last_name)
    first_name = input('Enter your given name: ')
    print('Given name:  ' + first_name)
    input_tensor, target_tensor, inp_lang, targ_lang, max_length_inp, max_length_targ = load_sequence_data()
    # Creating training and validation sets using an 80-20 split
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)
    print(len(input_tensor_train))
    print(len(target_tensor_train))
    print(len(input_tensor_val))
    print(len(target_tensor_val))

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 64
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    optimizer = tf.train.AdamOptimizer()
    # checkpoint_dir = './training_checkpoints'
    checkpoint_dir = './save_model'
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                     encoder=encoder,
                                     decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    job = translate(last_name, first_name, encoder, decoder, inp_lang, targ_lang, max_length_inp, max_length_targ)
    switch_to(GRAPH_MODE)
    print('predicted job: ', job[:-6])
    salary_model = load_model('save_model/model_job_to_salary.h5')
    X1 = np.array([job[:-6]])
    word_to_vec_map = load_glove_file()
    X1_train_embedding = np.array([sentence_to_avg(x, word_to_vec_map) for x in X1])
    print('Predicted salary: ', salary_model.predict(X1_train_embedding)[0])

if __name__ == '__main__':
    main()