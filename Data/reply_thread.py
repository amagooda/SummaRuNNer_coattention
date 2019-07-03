# from Data.reply_obj import ReplyObj


class ReplyThread(object):
    def __init__(self, post_id, title, initial_post, owner, summary_1, summary_2, selected_sentences):
        self.post_id = post_id
        self.title = title

        self.initial_post = initial_post
        self.initial_post_translated = []

        self.reply_sentences = []
        self.reply_sentences_translated = []

        self.selected_sentences = selected_sentences
        self.selected_sentences_translated = []

        self.summary_1 = summary_1
        self.summary_2 = summary_2


    def add_reply(self, comment_sentences):
        self.reply_sentences += comment_sentences
