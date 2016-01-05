from wordcloud import WordCloud
from itertools import repeat


def plot_clouds(model_word_weights_dict):
    for congress in model_word_weights_dict.keys():
        for idx, tup_l in enumerate(model_word_weights_dict[congress]):
            words = map(lambda x: list(repeat(x[0], int(x[1]))), tup_l)
            words = [word for sublist in words for word in sublist]
            text = ' '.join(words)
            wc = WordCloud(background_color = 'white', width = 800, height = 1800).generate(text)
            plt.imshow(wc)
            plt.figure(figsize = (9, 6))
            plt.axis('off')
            plt.title('Congress {} Main Topics {}'.format(congress, idx+1))
            plt.savefig('/Volumes/scsherm/Documents/Congress_work/plots/Congress_{}_{}'.format(congress, idx+1), format = 'png') 
            plt.close()


if __name__ == '__main__':
    plot_clouds(model_word_weights_dict)
    plt.show()
