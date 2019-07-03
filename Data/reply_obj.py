
class ReplyObj(object):
    def __init__(self, comment_sentences, selected_sentences):
        self.comment_sentences = comment_sentences
        self.selected_sentences = selected_sentences
        self.selected_indcies = selected_sentences