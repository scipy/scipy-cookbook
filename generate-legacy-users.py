"""
generate-legacy-users.py WIKI_PAGES_PATH

"""
from __future__ import absolute_import, print_function, division

import os
import re
import sys
from HTMLParser import HTMLParser


def main():
    path = sys.argv[1]

    users = {}
    pages = {}
    
    # Gather authors
    for root, dirs, files in os.walk(path):
        for d in dirs:
            revs = os.path.join(root, d, 'revisions')
            edit_log = os.path.join(root, d, 'edit-log')

            if not os.path.isdir(revs):
                continue

            if not os.path.isfile(edit_log):
                continue

            with open(edit_log, 'r') as handle:
                log_text = handle.read().rstrip()
                log_items = [x.split() for x in log_text.splitlines()]

            for fn in os.listdir(revs):
                fn = os.path.join(revs, fn)
                with open(fn, 'r') as handle:
                    r_text = handle.read()

                if ('CategoryHomepage' in r_text or 'home page' in r_text):
                    # User definition
                    for item in log_items:
                        if len(item) > 6:
                            if item[3] in ('About_SciPy', 'SciPy', 'Cookbook(2f)MayaVi(2f)tvtk'):
                                continue
                            users[item[6]] = item[3]
                            break
                    break

            if 'Cookbook' in d or 'PerformancePython' in d or 'ParallelProgramming' in d:
                for item in log_items:
                    pages.setdefault(d, []).append(item[6])

    # Load predefined users
    users['1273234778.27.13541'] = 'arjen'
    users['1181049059.11.16046'] = 'WarrenWeckesser'
    users['1232509635.1.1790'] = 'WarrenWeckesser'
    users['1143464513.17.11899'] = 'GaelVaroquaux'
    users['1359829272.72.54252'] = 'FrankBreitling'
    users['1196968472.52.21357'] = 'jesrl'
    users['1310512145.5.35406'] = 'RalphMoore'
    users['1134987132.31.5715'] = 'AndrewStraw'
    users['1283944978.14.25260'] = 'UnuTbu'
    users['1143464513.17.11899'] = 'GaelVaroquaux'
    users['1150066934.85.44238'] = 'FredericPetit'
    users['1157157190.0.28500'] = 'AMArchibald'
    users['1193155369.79.45281'] = 'Elby'
    users['1162990926.75.41968'] = 'PauliVirtanen'
    users['1144823769.21.43377'] = 'AngusMcMorland'
    users['1199025820.05.62034'] = 'TimMichelsen'
    users['1165998335.9.59069'] = 'MartinSpacek'
    users['1169591527.88.61566'] = 'MattKnox'
    users['1278911090.12.12663'] = 'ChristopherCampo'
    users['1230492524.42.55666'] = 'nokfi'
    users['1166654035.38.11968'] = 'VincentNijs'
    users['1160664185.24.177'] = 'NeilMB'
    users['1148241299.31.23452'] = 'GabrielGellner'
    users['1143248516.72.17557'] = 'FrancescAltet'
    users['1138755498.13.1844'] = 'BillBaxter'
    users['1138639075.54.47297'] = 'jh'
    users['1135217126.43.265'] = 'FernandoPerez'
    users['1228612570.79.23812'] = 'EgorZindy'
    users['1166684071.04.43914'] = 'ScottSinclair'
    users['1153060908.53.58092'] = 'EmmanuelleGouillart'
    users['1152996811.03.49324'] = 'NickFotopoulos'
    users['1135013651.92.25239'] = 'PearuPeterson'
    users['1263714477.79.46523'] = 'newacct'
    users['1321067029.14.1791'] = 'KristjanOnu'
    users['1244315014.64.10666'] = 'IvoMaljevic'
    users['1342900640.41.32910'] = 'thomas.haslwanter'
    users['1138834037.11.63568'] = 'TimCera'
    users['1306523623.53.4799'] = 'DmitriyRybalkin'
    users['1316810730.93.46683'] = 'TimSwast'
    users['1294906831.24.3474'] = 'MikeToews'
    users['1259530275.5.20672'] = 'JorgeEduardoCardona'
    users['1254476605.52.59655'] = 'wolfganglechner'
    users['1220051786.85.3734'] = 'SimonHook'
    users['1321851999.81.53674'] = 'BAlexRobinson'
    users['1245975199.53.27497'] = 'DavidPowell'
    users['1277317890.88.15794'] = 'AlanLue'
    users['1249699417.54.61063'] = 'mauro'
    users['1151666835.94.32020'] = 'WilliamHunter'
    users['1209753612.57.31138'] = 'JamesNagel'
    users['1241897483.76.24144'] = 'DatChu'
    users['1245526844.29.46176'] = 'RalfGommers'
    users['1312558832.94.40303'] = 'Pierre_deBuyl'
    users['1205277370.55.64453'] = 'keflavich'
    users['1147324201.78.18433'] = 'MichaelMcNeilForbes'
    users['1139447249.42.46498'] = 'RobManagan'
    users['1246487580.75.24764'] = 'MarshallPerrin'
    users['1340544644.02.6056'] = 'WesTurner'

    # Print results
    unknowns = {}
    page_uid = {}

    unknown_counter = 1
    unknown_names = {}

    for page, uids in sorted(pages.items()):
        editors = []
        seen = set()
        for uid in uids:
            if uid not in users:
                unknowns.setdefault(uid, 0)
                unknowns[uid] += 1

            if uid in seen:
                continue

            seen.add(uid)
            user = users.get(uid, 'unknown')
            if user == 'unknown':
                if uid not in unknown_names:
                    unknown_names[uid] = "Unknown[{0}]".format(unknown_counter)
                    unknown_counter += 1
                user = unknown_names[uid]
            editors.append(user)

        if page != 'Cookbook(2f)MayaVi(2f)examples':
            page_uid[uids[-1]] = page

        page = page.replace('(2f)', '/')
        page = page.replace('Cookbook/', '')
        page = page.replace('/', '_')
        print("{0}: {1}".format(page, ", ".join(editors)))

    # Sort by unknown
    items = sorted(unknowns.items(), key=lambda x: (x[1], x), reverse=True)
    for uid, count in items:
        print(unknown_names[uid], ":", uid, count, page_uid.get(uid, ''))


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


class StringMatcher(object):
    def __init__(self, items):
        self.fuzzyset = fuzzyset.FuzzySet(gram_size_lower=3,
                                          gram_size_upper=5)

        for item in sorted(items):
            self.fuzzyset.add(item)

    def get(self, item):
        r = []

        for fmt in [normalize, splitsub]:
            x = fmt(item)
            if x:
                q = self.fuzzyset.get(x)
                if q is not None:
                    r += q

        r.sort(key=lambda x: -x[0])
        if r:
            score, r = r[0]
            return self.aliases[r], score
        else:
            return None, 0


if __name__ == "__main__":
    main()
