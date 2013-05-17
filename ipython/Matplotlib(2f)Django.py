# <markdowncell>

# 1.  1.  Please edit system and help pages ONLY in the moinmaster wiki!
#         For more
#     2.  information, please see MoinMaster:MoinPagesEditorGroup.
#     3.  master-page:Unknown-Page
#     4.  master-date:Unknown-Date
#     5.  acl MoinPagesEditorGroup:read,write,delete,revert All:read
# 
# Using MatPlotLib to dynamically generate charts in a Django web service
# -----------------------------------------------------------------------
# 
# You need to have a working Django installation, plus matplotlib.
# 
# ### Example 1 - PIL Buffer
# 
# <codecell>


# file charts.py
def simple(request):
    import random
    import django
    import datetime
    
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.dates import DateFormatter

    fig=Figure()
    ax=fig.add_subplot(111)
    x=[]
    y=[]
    now=datetime.datetime.now()
    delta=datetime.timedelta(days=1)
    for i in range(10):
        x.append(now)
        now+=delta
        y.append(random.randint(0, 1000))
    ax.plot_date(x, y, '-')
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    canvas=FigureCanvas(fig)
    response=django.http.HttpResponse(content_type='image/png')
    canvas.print_png(response)
    return response

# <markdowncell>

# Since some versions of Internet Explorer ignore the content\_type. The
# URL should end with ".png". You can create an entry in your urls.py like
# this:
# 
# <codecell>


    ...
    (r'^charts/simple.png$', 'myapp.views.charts.simple'),
    ...

# <markdowncell>

# * * * * *
# 
# `CategoryCookbookMatplotlib`
# 