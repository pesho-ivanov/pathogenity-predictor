"""Q8: render a dated, cited survey of variant-effect prediction tools.

The authored catalog is bundled with this module. No variants, models or network
services are accessed; the notebook reproduces the reviewed survey offline.
"""

from datetime import date
import hashlib
import html
import importlib.metadata
import json
from pathlib import Path
import platform
from urllib.parse import urlparse

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CATALOG = Path(__file__).with_name('q8_catalog.json')
OUTPUT = ROOT / 'notebooks/results/q8'
TARGETS = ['Broad SNV score', 'Missense impact', 'Splicing effects', 'Regulatory effects']
COLORS = ['#31688e', '#28866b', '#ba7700', '#8061a8']


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_survey(survey):
    """Fail on missing provenance or inconsistent catalog/figure membership."""
    reviewed = date.fromisoformat(survey['reviewed_on'])
    names = [tool['name'] for tool in survey['tools']]
    if not names or len(names) != len(set(names)):
        raise ValueError('Tool names must be present and unique')
    required = ['release', 'method', 'variants', 'inputs', 'access', 'compute',
                'license', 'training', 'clinvar', 'caution']
    used_sources = set()
    for tool in survey['tools']:
        if any(not isinstance(tool.get(key), str) or not tool[key].strip() for key in required):
            raise ValueError(f'Missing comparison information: {tool["name"]}')
        if not tool['targets'] or set(tool['targets']) - set(TARGETS):
            raise ValueError(f'Invalid target category: {tool["name"]}')
        if tool['reviewed_on'] != reviewed.isoformat():
            raise ValueError('Tool review dates must match the survey snapshot')
        if not tool['sources'] or set(tool['sources']) - survey['sources'].keys():
            raise ValueError(f'Missing source attribution: {tool["name"]}')
        used_sources.update(tool['sources'])
        revision = tool['source_revision']
        if revision:
            sha = revision['revision']
            if len(sha) != 40 or set(sha) - set('0123456789abcdef'):
                raise ValueError('Source revision must be a full Git commit')
            if date.fromisoformat(revision['commit_date'][:10]) > reviewed:
                raise ValueError('Source commit is later than the review date')
    for key, source in survey['sources'].items():
        url = urlparse(source['url'])
        if (not source['title'] or url.scheme != 'https' or not url.netloc
                or source['reviewed_on'] != reviewed.isoformat()):
            raise ValueError(f'Invalid source metadata: {key}')
    if used_sources != survey['sources'].keys():
        raise ValueError('The bibliography must contain exactly the cited sources')
    for recommendation in survey['shortlist']:
        if not recommendation['tools'] or set(recommendation['tools']) - set(names):
            raise ValueError('Shortlist refers to a tool absent from the survey')
    return {
        'unique_tools_and_known_targets': True,
        'comparison_fields_and_citations_complete': True,
        'source_dates_and_revisions_consistent': True,
        'shortlist_membership_valid': True,
        'scope': 'Catalog consistency only; no biological leakage or accuracy claim.',
    }


def load_survey():
    survey = json.loads(CATALOG.read_text())
    check_survey(survey)
    print(f'{len(survey["tools"])} representative tools · reviewed {survey["reviewed_on"]} · CPU only · offline')
    print('Reproduces the bundled review; no random sampling, inference or benchmark.')
    return survey


def source_links(survey, identifiers):
    return ' · '.join(
        f'<a href="{html.escape(survey["sources"][key]["url"], quote=True)}">'
        f'{html.escape(survey["sources"][key]["title"])}</a>' for key in identifiers
    )


def details(title, body):
    from IPython.display import HTML, display
    display(HTML(f'<details><summary>{html.escape(title)}</summary>'
                 f'<div style="overflow-x:auto; margin-top:0.6em">{body}</div></details>'))


def show_landscape(survey):
    """Map documented scoring targets; this is not measured variant coverage."""
    check_survey(survey)
    tools = survey['tools']
    fig, ax = plt.subplots(figsize=(9, 6.3), layout='constrained')
    for y, tool in enumerate(tools):
        for x, target in enumerate(TARGETS):
            selected = target in tool['targets']
            ax.scatter(x, y, s=300, marker='s',
                       color=COLORS[x] if selected else '#edf0f2', edgecolor='white')
    ax.set(xticks=range(len(TARGETS)), xticklabels=[s.replace(' ', '\n', 1) for s in TARGETS],
           yticks=range(len(tools)), yticklabels=[tool['name'] for tool in tools],
           xlim=(-.65, 3.65), ylim=(len(tools) - .4, -.6))
    ax.tick_params(length=0, labelsize=10)
    ax.xaxis.tick_top()
    ax.set_title('Different tools answer different biological questions', loc='left', pad=48, fontweight='bold')
    for spine in ax.spines.values():
        spine.set_visible(False)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / 'tool_landscape.png', dpi=160)
    plt.show()
    plt.close(fig)


def show_comparison(survey):
    check_survey(survey)
    for title, columns in [
        ('Compare releases, scope, inputs, availability and compute',
         ['release', 'method', 'variants', 'inputs', 'access', 'compute', 'license']),
        ('Audit training, ClinVar exposure and remaining limitations',
         ['training', 'clinvar', 'caution']),
    ]:
        rows = []
        for tool in survey['tools']:
            row = {'Tool': html.escape(tool['name'])}
            row.update({key.title(): html.escape(tool[key]) for key in columns})
            row['Sources'] = source_links(survey, tool['sources'])
            rows.append(row)
        table = pd.DataFrame(rows).to_html(index=False, escape=False, border=0)
        details(title, table)
    sources = ''.join(f'<li>{source_links(survey, [key])}</li>' for key in survey['sources'])
    metadata = {key: survey[key] for key in ['reviewed_on', 'scope', 'method', 'version_policy', 'compute_policy']}
    metadata['source_revisions'] = {tool['name']: tool['source_revision'] for tool in survey['tools'] if tool['source_revision']}
    details('Review method, pinned references and bibliography',
            '<pre style="white-space:pre-wrap">' + html.escape(json.dumps(metadata, indent=2))
            + '</pre><ol>' + sources + '</ol>')


def show_shortlist(survey):
    """Present explicit design judgments separately from the sourced facts."""
    import textwrap
    from IPython.display import Markdown, display
    check_survey(survey)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 5.8), layout='constrained')
    for ax, item, color in zip(axes.flat, survey['shortlist'], COLORS):
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.axis('off')
        ax.add_patch(FancyBboxPatch((.02, .04), .94, .91, boxstyle='round,pad=.02',
                                   facecolor='#f5f7f9', edgecolor=color, linewidth=2))
        ax.text(.07, .85, item['scope'], color=color, fontsize=12, fontweight='bold', va='top')
        ax.text(.07, .67, ' / '.join(item['tools']), fontsize=13, fontweight='bold', va='top')
        ax.text(.07, .48, textwrap.fill(item['reason'], width=49), fontsize=10, va='top', linespacing=1.4)
    fig.suptitle('Candidate baselines by scope · design judgment, not measured superiority', fontsize=13)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / 'baseline_shortlist.png', dpi=160)
    plt.show()
    plt.close(fig)
    rules = ''.join('<li>' + html.escape(rule) + '</li>' for rule in survey['evaluation_rules'])
    details('Rules for a later benchmark on Q1 partitions', '<ol>' + rules + '</ol>')
    display(Markdown('Longer-context models require new overlap checks; Q1’s **1,024-base** grouping is insufficient to certify their contexts.'))


def export_results(survey):
    checks = check_survey(survey)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'survey.json').write_text(json.dumps(survey, indent=2, ensure_ascii=False) + '\n')
    rows = [{**tool, 'targets': '; '.join(tool['targets']),
             'sources': '; '.join(survey['sources'][key]['url'] for key in tool['sources']),
             'source_revision': json.dumps(tool['source_revision'])} for tool in survey['tools']]
    pd.DataFrame(rows).to_csv(OUTPUT / 'tool_comparison.csv', index=False)
    files = ['survey.json', 'tool_comparison.csv', 'tool_landscape.png', 'baseline_shortlist.png']
    provenance = {
        'reviewed_on': survey['reviewed_on'], 'catalog_sha256': digest_file(CATALOG),
        'implementation_sha256': digest_file(__file__),
        'python': platform.python_version(),
        'packages': {name: importlib.metadata.version(name) for name in ['pandas', 'matplotlib', 'ipython']},
        'random_seed': None, 'compute': 'CPU only; no model weights or external datasets needed',
        'variants_read': False, 'models_fitted_or_scored': False, 'network_required': False,
        'checks': checks,
        'artifacts': {name: digest_file(OUTPUT / name) for name in files},
    }
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Survey checks passed; catalog, comparisons, figures and provenance saved to results/q8/.')


def show_conclusion(survey):
    from IPython.display import Markdown, display
    display(Markdown(survey['conclusion']))
