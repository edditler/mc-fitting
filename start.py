import ipywidgets as ipw


def get_start_widget(appbase, jupbase):
    template = """
    <table>
    <tr>
    <td valign="top"><ul>
    <li><a href="{appbase}/geo_opt.ipynb" target="_blank">Geometry Optimization</a>
    <li><a href="{appbase}/circus.ipynb" target="_blank">Circus</a>
    <li><a href="{appbase}/energy_forces.ipynb" target="_blank">Forces</a>
    <li><a href="{appbase}/eam.ipynb" target="_blank">EAM</a>
    </ul></td>
    </tr></table>
    """
    html = template.format(appbase=appbase, jupbase=jupbase)
    return ipw.HTML(html)

