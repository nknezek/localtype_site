import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO




def colored_score(exp, cityid):
    low = 30
    high = 70
    colors = ['#a50026', '#ffc107', '#006837']
    score = exp.predict_proba[cityid] * 100
    if score < low:
        color = colors[0]
        text = '{:.0f}% - poor'.format(score)
    elif score >= low and score < high:
        color = colors[1]
        text = '{:.0f}% - acceptable'.format(score)
    else:
        color = colors[2]
        text = '{:.0f}% - great!'.format(score)
    scoretxt = "<strong><span style='color: {}'>{}</span></strong>".format(color, text)
    return scoretxt


def plot_top_words(exp, N=6):
    (_,idscores),= exp.as_map().items()
    Nplt = min(len(idscores),N)
    namescores = exp.domain_mapper.map_exp_ids(idscores)
    names = [x[0] for x in namescores][:Nplt]
    scores = [x[1] for x in namescores][:Nplt]
    y = range(Nplt,0,-1)

    fig, ax = plt.subplots(figsize=(2.5,4))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    # ax.axis('off')
    colors = plt.cm.RdYlGn((np.sign(scores)+1)/2)
    plt.barh(y,scores,color=colors)
    # plt.title("Influential Words")
    plt.yticks(y, names, fontsize=16)
    plt.tight_layout()

    try:
        figfile = BytesIO()
        plt.savefig(figfile, format='svg')
        figfile.seek(0)
        figdata_svg = b'<svg' + figfile.getvalue().split(b'<svg')[1]
        return figdata_svg.decode('utf-8')
    except:
        return None
    # return mpld3.fig_to_html(fig)

def plot_cityscores(exp,cityid, N=6):
    cn = exp.class_names
    p = exp.predict_proba
    Nplt = min(N,len(cn))
    cns = [x[1] for x in sorted(zip(p,cn))][:Nplt]
    ps = np.sort(p)[:Nplt]
    inds = np.argsort(p)[:Nplt]
    colors = plt.cm.RdYlGn(np.zeros(len(ps)))
    if cityid in inds:
        colors[np.where(inds==cityid)] = plt.cm.RdYlGn(1.)
    y = range(0,Nplt)
    fig, ax = plt.subplots(figsize=(2.5,4))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.barh(y,ps,color=colors)
    plt.yticks(y,cns)
    plt.title('Top Cities')
    plt.tight_layout()

    try:
        figfile = BytesIO()
        plt.savefig(figfile, format='svg')
        figfile.seek(0)
        figdata_svg = b'<svg' + figfile.getvalue().split(b'<svg')[1]
        return figdata_svg.decode('utf-8')
    except:
        return None
    # return mpld3.fig_to_html(fig)

def color_words(exp):
    words = exp.domain_mapper.indexed_string.as_list
    positions = exp.domain_mapper.indexed_string.positions

    pos_inds = []
    pos_alphas = []
    neg_inds = []
    neg_alphas = []
    for k,v in exp.local_exp[exp.available_labels()[0]]:
        for p in positions[k]:
            if v>0:
                pos_inds.append(p)
                pos_alphas.append(np.abs(v))
            else:
                neg_inds.append(p)
                neg_alphas.append(np.abs(v))

    html_text = '<p>'
    for i,w in enumerate(words):
        if i in pos_inds:
            html_text += "<span style='color: #006837; font-weight:bold;'>"+w+"</span>"
        elif i in neg_inds:
            html_text += "<span style='color: #a50026; font-weight:bold;'>"+w+"</span>"
        else:
            html_text += w
    html_text += '</p>'
    return html_text


def list_cities(exp,cityid,N=6,):
    cnames = list(exp.class_names)
    p = list(exp.predict_proba)
    Nplt = min(N,len(cnames))
    cns = [x[1] for x in sorted(zip(p,cnames), reverse=True)]
    ps = sorted(p, reverse=True)
    all_inds = np.argsort(p)[::-1]
    inds = all_inds[:Nplt]
    if cityid not in inds:
        Nplt = Nplt - 1
    html_cities = ''
    for i,(ind,name) in enumerate(zip(inds[:Nplt],cns[:Nplt])):
        if ind == cityid:
            html_cities += '<tr><th><strong> {}. {} </strong></th>'.format(i+1,name)
        else:
            html_cities += '<tr><th> {}. {} </th></tr>'.format(i+1,name)
    if cityid not in inds:
        rank = np.where(all_inds==cityid)[0][0]
        html_cities += '<tr><th>:</th></tr> <tr><th><strong> {}. {} </strong></th></tr>'.format(rank+1,cnames[cityid])
    return html_cities