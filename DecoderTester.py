import logging


class Test:
    def __init__(self, documents):
        self.documents = documents
        self.correct_states = 0
        self.total_states = sum([len(document) for document in documents])
        self.correct_documents = 0
        self.total_documents = len(documents)

    def test_decoder(self, decoder, lowercase=False):
        self.correct_states = 0
        self.correct_documents = 0
        for i, document in enumerate(self.documents):
            logging.info("Decoder test for document %s", i)
            observations = []
            true_states = []
            for observation, state in document:
                observations.append(observation)
                true_states.append(state)
            if lowercase:
                observations = [observation.lower() for observation in observations]
            decoded_sequence = decoder.decode(observations)
            if decoded_sequence == true_states:
                self.correct_documents += 1
            for i, decoded_state in enumerate(decoded_sequence):
                if decoded_state == true_states[i]:
                    self.correct_states += 1
        return self.correct_states/self.total_states, self.correct_documents/self.total_documents
