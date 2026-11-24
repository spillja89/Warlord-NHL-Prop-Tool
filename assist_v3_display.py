"""Only strongest matched optional badge per family; exact routes stay in Inspect."""
from html import escape

def badges(moves):
    from move_return_grades import grade_rank
    from assist_return_monitor import display_stats
    groups={}
    for m in moves:
        if not m.get('assist_v3') or not m.get('optional_buff'):continue
        family=m['display_family'];prior=groups.get(family)
        if prior is None or (grade_rank(m) or ())>(grade_rank(prior) or ()):groups[family]=m
    if not groups:return ''
    items=[]
    for m in groups.values():
        s=display_stats(m);label=m['name'];parent=m.get('parent_name') or 'Original .5 Green Conf81 baseline'
        tip=parent+' + '+str(m['additional_condition'])+'; optional exact-parent record, not tonight’s probability. Full/recent evidence in Inspect.'
        items.append('<span class="wn-goalie-buff" title="'+escape(tip,quote=True)+'">✦ '+escape(label)+f" <b>{s['hits']}/{s['plays']} · {100*s['hits']/s['plays']:.1f}%</b></span>")
    return '<div class="wn-v3-buffs">'+''.join(items)+'</div>'
