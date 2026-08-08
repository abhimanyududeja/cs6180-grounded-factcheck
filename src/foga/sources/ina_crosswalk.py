"""INA section <-> 8 U.S.C. section crosswalk.

Practitioners, USCIS, and the FAM all cite the Immigration and Nationality Act
by its *act* section number ("INA 214(b)"), but the codified text published by
the Office of the Law Revision Counsel is numbered by its *U.S. Code* section
("8 U.S.C. 1184(b)"). A user asking about "INA 245(i)" will never match text
labeled "8 U.S.C. 1255" on lexical overlap alone.

So every statute Document gets BOTH citations, and the chunk text is prefixed
with both forms. This single mapping is responsible for a large share of the
retrieval quality on statute questions.

Coverage: the principal operative provisions of INA Titles I-III. Sections not
listed fall back to the U.S. Code citation alone, which is still correct — just
less discoverable.
"""

from __future__ import annotations

# INA section -> 8 U.S.C. section (both as strings, since suffixes like "274A"
# and "240B" are common).
INA_TO_USC: dict[str, str] = {
    # Title I — General provisions
    "101": "1101", "102": "1102", "103": "1103", "104": "1104", "105": "1105",
    "106": "1105a",
    # Title II ch.1 — Selection system
    "201": "1151", "202": "1152", "203": "1153", "204": "1154", "205": "1155",
    "206": "1156", "207": "1157", "208": "1158", "209": "1159", "210": "1160",
    "210A": "1161",
    # Title II ch.2 — Admission qualifications
    "211": "1181", "212": "1182", "213": "1183", "213A": "1183a", "214": "1184",
    "215": "1185", "216": "1186a", "216A": "1186b", "217": "1187", "218": "1188",
    "219": "1189",
    # Title II ch.3 — Visas
    "221": "1201", "222": "1202", "223": "1203", "224": "1204",
    # Title II ch.4 — Inspection, removal
    "231": "1221", "232": "1222", "233": "1223", "234": "1224", "235": "1225",
    "235A": "1225a", "236": "1226", "236A": "1226a", "237": "1227", "238": "1228",
    "239": "1229", "240": "1229a", "240A": "1229b", "240B": "1229c", "240C": "1230",
    "241": "1231", "242": "1252", "243": "1253", "244": "1254a",
    # Title II ch.5 — Adjustment of status
    "245": "1255", "245A": "1255a", "246": "1256", "247": "1257", "248": "1258",
    "249": "1259", "250": "1260",
    # Title II ch.8 — Offenses / employment verification
    "271": "1321", "272": "1322", "273": "1323", "274": "1324", "274A": "1324a",
    "274B": "1324b", "274C": "1324c", "275": "1325", "276": "1326", "277": "1327",
    "278": "1328", "279": "1329", "280": "1330",
    # Title II ch.9 — Administration
    "281": "1351", "282": "1352", "283": "1353", "284": "1354", "285": "1355",
    "286": "1356", "287": "1357", "288": "1358", "289": "1359", "290": "1360",
    "291": "1361", "292": "1362", "293": "1363", "294": "1363a",
    # Title III ch.1 — Nationality at birth
    "301": "1401", "302": "1402", "303": "1403", "304": "1404", "305": "1405",
    "306": "1406", "307": "1407", "308": "1408", "309": "1409",
    # Title III ch.2 — Naturalization
    "310": "1421", "311": "1422", "312": "1423", "313": "1424", "314": "1425",
    "315": "1426", "316": "1427", "317": "1428", "318": "1429", "319": "1430",
    "320": "1431", "322": "1433", "324": "1435", "325": "1436", "326": "1437",
    "327": "1438", "328": "1439", "329": "1440", "330": "1441", "331": "1442",
    "332": "1443", "333": "1444", "334": "1445", "335": "1446", "336": "1447",
    "337": "1448", "338": "1449", "339": "1450", "340": "1451", "341": "1452",
    "342": "1453", "343": "1454", "344": "1455", "346": "1457", "347": "1458",
    "348": "1459",
    # Title III ch.3 — Loss of nationality
    "349": "1481", "350": "1482", "351": "1483", "352": "1484", "353": "1485",
    "354": "1486", "355": "1487", "356": "1488", "357": "1489",
    "358": "1501", "359": "1502", "360": "1503", "361": "1504",
}

USC_TO_INA: dict[str, str] = {v: k for k, v in INA_TO_USC.items()}


def ina_for_usc(usc_section: str) -> str | None:
    """'1184' -> '214'. Accepts '1184' or '§ 1184'."""
    key = usc_section.strip().lstrip("§ ").strip()
    return USC_TO_INA.get(key)


def usc_for_ina(ina_section: str) -> str | None:
    """'214' -> '1184'. Accepts '214', 'INA 214', '214(b)'."""
    key = ina_section.upper().replace("INA", "").strip().lstrip("§ ").strip()
    key = key.split("(")[0].strip()
    return INA_TO_USC.get(key)
