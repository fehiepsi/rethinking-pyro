{%- extends "basic.tpl" -%}

{% block input_group -%}
{%- if cell.metadata.hide_input or nb.metadata.hide_input -%}
{%- else -%}
    {{ super() }}
{%- endif -%}
{% endblock input_group %}

{% block output_group -%}
{%- if cell.metadata.hide_output -%}
{%- else -%}
    {{ super() }}
{%- endif -%}
{% endblock output_group %}

{% block markdowncell %}
{%- if cell.metadata.navigation -%}
    <div class="cell border-box-sizing text_cell rendered">
    <div class="inner_cell">
    <div class="text_cell_render border-box-sizing rendered_html">
    <div class="navigation">
    {{ cell.source  | markdown2html | strip_files_prefix }}
    </div>
    </div>
    </div>
    </div>
{%- else -%}
    {{ super() }}
{%- endif -%}
{%- endblock markdowncell %}
