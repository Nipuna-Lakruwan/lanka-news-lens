"""Weak topic labels for Sri Lankan news from two sources.

1. **URL section** where an outlet puts one in its URL and the section maps
   cleanly to a topic (for example ``ada.lk/sports/...`` gives ``sports``).
2. **Trilingual keyword rules** on the title (weight 2) and the first 300
   characters of the body (weight 1). Sinhala and Tamil rules are *substring*
   stems, because both languages attach case endings and particles to the
   stem (``අත්අඩංගුවට``, ``கைதானார்``). English rules are whole words.

An article gets a keyword label only when the best topic scores at least 3 and
beats the runner-up by at least 2. Everything else stays unlabelled. These
labels are *training* signal only. The test set is a separate random sample
labelled by hand (see ``scripts/make_testset.py``).
"""
from __future__ import annotations

import re

TOPICS = ["politics", "economy", "crime", "disaster_weather", "health", "sports",
          "education", "international", "other"]

SECTION_LABELS = {
    ("adalk", "business"): "economy",
    ("adalk", "sports"): "sports",
    ("adalk", "international"): "international",
    ("adalk", "technology"): "other",
    ("adalk", "religion"): "other",
    ("dailymirrorlk", "international"): "international",
    ("dailymirrorlk", "business-news"): "economy",
    ("dailymirrorlk", "business"): "economy",
    ("dailymirrorlk", "business-main"): "economy",
    ("tamilmirrorlk", "உலக-செய்திகள்"): "international",
    ("tamilmirrorlk", "பிரதான-விளையாட்டு"): "sports",
    ("tamilmirrorlk", "விளையாட்டு"): "sports",
    ("tamilmirrorlk", "cinema"): "other",
    ("tamilmirrorlk", "விநோத-உலகம்"): "other",
    ("lankadeepalk", "world_news"): "international",
    ("lankadeepalk", "politics"): "politics",
}

# --- keyword rules --------------------------------------------------------
EN = {
    "politics": "parliament|parliamentary|president|presidential|prime minister|minister|ministers|cabinet|mp|mps|"
                "election|elections|poll|polls|vote|voting|npp|sjb|slpp|unp|jvp|itak|tna|opposition|govt|"
                "government|constitution|amendment|speaker|provincial council|local government|anura|"
                "dissanayake|ranil|wickremesinghe|sajith|premadasa|rajapaksa|namal|harini|amarasuriya|party|"
                "campaign|bill|legislation|governor|diplomatic|ambassador",
    "economy": "economy|economic|imf|inflation|rupee|dollar|cbsl|central bank|budget|tax|taxes|vat|revenue|"
               "debt|bond|bonds|treasury|gdp|export|exports|import|imports|tariff|tariffs|tourism|tourist|"
               "tourists|investment|investor|investors|cse|stock|shares|aspi|bank|banks|banking|interest rate|"
               "fuel price|prices|price|electricity tariff|ceb|trade|company|profit|revenue|billion|million|"
               "loan|loans|reserves|remittances|business|industry|apparel|manufacturing",
    "crime": "arrested|arrest|arrests|police|shot|shooting|murder|murdered|killed|stabbed|robbery|theft|"
             "stolen|heroin|ice|cannabis|kerala ganja|drug|drugs|narcotics|smuggling|smuggled|cid|court|"
             "remanded|remand|bail|sentenced|indicted|fraud|bribery|bribe|corruption|ciaboc|suspect|suspects|"
             "detained|custody|underworld|gang|assault|raped|rape|abuse|kidnapped|cybercrime|scam",
    "disaster_weather": "flood|floods|flooding|landslide|landslides|rain|rains|rainfall|heavy rain|cyclone|"
                        "storm|thunderstorm|lightning|weather|met department|meteorology|drought|tsunami|"
                        "earthquake|dmc|disaster|disasters|inundated|monsoon|wind|winds|spill gates|"
                        "evacuated|ditwah|heatwave|heat",
    "health": "health|hospital|hospitals|doctor|doctors|gmoa|medical|medicine|medicines|dengue|covid|"
              "virus|disease|diseases|patients|patient|vaccine|vaccination|cancer|nurses|surgery|world health|"
              "chikungunya|influenza|outbreak|kidney|mental health|nutrition|pharmaceutical|nmra",
    "sports": "cricket|match|matches|odi|t20|t20i|test series|world cup|tournament|wicket|wickets|"
              "innings|batsman|batter|bowler|football|rugby|athletics|olympic|olympics|asia cup|ipl|lpl|"
              "squad|captain|coach|championship|medal|volleyball|netball|hockey|tennis|marathon|slc|"
              "sanath|mendis|hasaranga|asalanka|wanindu|nissanka|kusal",
    "education": "school|schools|student|students|teacher|teachers|university|universities|ugc|"
                 "examination|examinations|exam|exams|a/l|o/l|grade 5|scholarship|education|"
                 "undergraduates|undergraduate|curriculum|principal|principals|tuition|degree|campus|"
                 "lecturers|futa|graduates|education reforms",
    "international": "israel|gaza|hamas|iran|ukraine|russia|putin|trump|biden|harris|white house|pakistan|"
                     "bangladesh|nepal|maldives|myanmar|afghanistan|china's|beijing|washington|"
                     "lebanon|hezbollah|syria|yemen|houthi|europe|eu|nato|un general assembly|modi|"
                     "new delhi|tamil nadu|japan|korea|taiwan|philippines|indonesia|saudi|uae|qatar|"
                     "britain|london|france|germany|canada|australia|united states|u.s.",
    "other": "film|movie|cinema|actor|actress|singer|song|music|concert|teledrama|vesak|poson|perahera|"
             "temple|church|kovil|mosque|religious|buddhist|thera|bishop|pope|festival|avurudu|new year|"
             "christmas|ramadan|deepavali|art|exhibition|culture|cultural|technology|ai|smartphone|"
             "app|whatsapp|facebook|google|apple|elephant|wildlife|leopard|environment|beauty pageant",
}

SI = {
    "politics": "පාර්ලිමේන්තු|ජනාධිපති|අගමැති|අග්‍රාමාත්‍ය|ඇමති|අමාත්‍ය|කැබිනට්|මන්ත්‍රී|මැතිවරණ|ඡන්ද|"
                "විපක්ෂ|ආණ්ඩුව|රජය|පක්ෂ|ජාතික ජන බලවේග|සමගි ජන|පොහොට්ටු|එජාප|ජවිපෙ|ආණ්ඩුක්‍රම|"
                "අනුර|රනිල්|සජිත්|රාජපක්ෂ|නාමල්|හරිනි|කථානායක|ආණ්ඩුකාර|පළාත් සභා|පළාත් පාලන|"
                "තානාපති|පනත|සංශෝධන|දේශපාලන",
    "economy": "ආර්ථික|ජාමූඅ|ඉන්ධන මිල|මිල|බදු|වැට්|රුපියල|ඩොලර|මහ බැංකු|අයවැය|ණය|උද්ධමන|"
               "අපනයන|ආනයන|සංචාරක|ආයෝජන|කොටස් වෙළඳපොළ|බැංකු|පොලී|විදුලි ගාස්තු|වෙළඳ|"
               "ව්‍යාපාර|කර්මාන්ත|තේ කර්මාන්ත|සහල්|වී මිල|ගොවි|ආදායම|සංචිත|බිලියන",
    "crime": "අත්අඩංගු|අල්ලයි|අල්ලා|වෙඩි|ඝාතන|මරා|පිහි|කොල්ල|සොරකම|සොරු|හෙරොයින්|අයිස්|"
             "ගංජා|කුඩු|මත්ද්‍රව්‍ය|ජාවාරම|සීඅයිඩී|අපරාධ|රක්ෂිත බන්ධන|ඇප|අධිකරණ|නඩු|මහේස්ත්‍රාත්|"
             "චෝදනා|වංචා|අල්ලස්|දූෂණ|සැකකරු|සැකකාර|පාතාල|පහරදී|ප්‍රහාර|දූෂණය|ස්ත්‍රී දූෂණ|පොලිසි|පොලීසි|"
             "බන්ධනාගාර|වරෙන්තු|දඩ",
    "disaster_weather": "ගංවතුර|නායයෑම|නාය යෑම|නායයාම|වැසි|වර්ෂාප|සුළි|කුණාටු|අකුණු|ගිගුරුම්|කාලගුණ|"
                        "නියඟ|සුනාමි|භූ කම්පන|භූමිකම්පා|ආපදා|ජලාශ|වාන් දොරටු|දිත්වා|සුළං|"
                        "උෂ්ණත්ව|ජලයෙන් යට",
    "health": "සෞඛ්‍ය|රෝහල|වෛද්‍ය|ඩෙංගු|කොවිඩ්|වෛරස|රෝග|රෝගී|එන්නත|පිළිකා|හෙද|සැත්කම්|ඖෂධ|"
              "බෙහෙත්|චිකුන්ගුන්යා|උණ රෝග|වසංගත|වකුගඩු|පෝෂණ|ලෝක සෞඛ්‍ය",
    "sports": "ක්‍රිකට්|තරග|එක්දින|ටී20|ටෙස්ට්|ලෝක කුසලාන|කඩුලු|ඉනිම|පිතිකරු|පන්දු යවන|පාපන්දු|"
              "රග්බි|මලල ක්‍රීඩා|ඔලිම්පික්|ආසියා කුසලාන|ක්‍රීඩා|පදක්කම|වොලිබෝල්|දැල්පන්දු|"
              "ශූරතා",
    "education": "පාසල්|පාසැල්|ශිෂ්‍ය|ගුරු|විශ්වවිද්‍යාල|විශ්ව විද්‍යාල|විභාග|උසස් පෙළ|සාමාන්‍ය පෙළ|"
                 "ශිෂ්‍යත්ව|අධ්‍යාපන|උපාධි|විදුහල්පති|ටියුෂන්|කථිකාචාර්ය|අධ්‍යාපන ප්‍රතිසංස්කරණ",
    "international": "ඊශ්‍රායල|ගාසා|හමාස්|ඉරාන|යුක්රේන|රුසියා|පුටින්|ට්‍රම්ප්|බයිඩන්|ඇමෙරිකා|අමෙරිකා|"
                     "පකිස්ථාන|බංග්ලාදේශ|නේපාල|මාලදිවයින|මියන්මාර|ඇෆ්ගනිස්ථාන|චීනය|ලෙබනන|සිරියා|"
                     "යුරෝපා|මෝදි|දිල්ලි|ජපාන|කොරියා|තායිවාන|සෞදි|එක්සත් රාජධානි|බ්‍රිතාන්‍ය|ප්‍රංශ|"
                     "ජර්මනි|කැනඩා|ඕස්ට්‍රේලියා|ඉන්දියාවේ",
    "other": "චිත්‍රපට|නළු|නිළි|ගායක|ගීත|සංගීත|ප්‍රසංග|ටෙලි|වෙසක්|පොසොන්|පෙරහැර|පෙරහර|විහාර|පන්සල|"
             "දේවස්ථාන|කෝවිල|පල්ලි|ආගමික|හිමියන්|ස්වාමීන්|උත්සව|අවුරුදු|නත්තල්|රාමසාන්|දීපවාලි|කලාකරු|කලා උළෙල|"
             "ප්‍රදර්ශන|සංස්කෘතික|තාක්ෂණ|කෘත්‍රිම බුද්ධි|ජංගම දුරකථන|වට්ස්ඇප්|ෆේස්බුක්|වන අලි|අලියා|වනජීවී|"
             "කොටියා|දිවියා|පරිසර|දන්සල|ඇදහිලි",
}

TA = {
    "politics": "நாடாளுமன்ற|பாராளுமன்ற|ஜனாதிபதி|பிரதமர்|அமைச்சர்|அமைச்சரவை|எம்.பி|உறுப்பினர்|தேர்தல்|"
                "வாக்கு|எதிர்க்கட்சி|அரசாங்க|கட்சி|தேசிய மக்கள் சக்தி|ஐக்கிய மக்கள் சக்தி|பொதுஜன பெரமுன|"
                "தமிழரசு|அனுர|ரணில்|சஜித்|ராஜபக்ஷ|நாமல்|ஹரிணி|சபாநாயகர்|ஆளுநர்|மாகாண சபை|"
                "உள்ளூராட்சி|அரசியலமைப்பு|திருத்தம்|தூதுவர்|சட்டமூல|அரசியல்",
    "economy": "பொருளாதார|சர்வதேச நாணய நிதிய|பணவீக்க|ரூபா|டொலர்|மத்திய வங்கி|வரவு செலவு|வரிகள்|வருமான வரி|வரி அறவீ|"
               "கடன்|ஏற்றுமதி|இறக்குமதி|சுற்றுலா|முதலீ|பங்குச் சந்தை|வங்கி|வட்டி|எரிபொருள் விலை|"
               "விலை|மின் கட்டண|வர்த்தக|வணிக|கைத்தொழில்|தேயிலை|தேங்காய்|அரிசி|நெல்|விவசாய|வருமான|"
               "பில்லியன்|மில்லியன்",
    "crime": "கைது|துப்பாக்கி|சூடு|கொலை|கொன்ற|கத்தி|கொள்ளை|திருட்டு|திருடிய|ஹெரோயின்|ஐஸ்|கஞ்சா|"
             "போதை|கடத்தல்|குற்ற|விளக்கமறியல்|பிணை|நீதிமன்ற|வழக்கு|நீதவான்|மோசடி|இலஞ்ச|லஞ்ச|ஊழல்|"
             "சந்தேகநபர்|சந்தேக நபர்|பாதாள|தாக்குதல்|தாக்கிய|பாலியல்|துஷ்பிரயோக|பொலிஸ்|பொலிஸார்|சிறைச்சாலை|"
             "பிடியாணை|தண்டனை|அபராத",
    "disaster_weather": "வெள்ள|மண்சரிவு|மழை|சூறாவளி|புயல்|இடியுடன்|மின்னல்|வானிலை|காலநிலை|வறட்சி|சுனாமி|"
                        "நிலநடுக்க|பூகம்ப|அனர்த்த|பேரிடர்|நீர்த்தேக்க|வான் கதவு|வெளியேற்ற|டித்வா|காற்று|"
                        "வெப்ப",
    "health": "சுகாதார|வைத்தியசாலை|மருத்துவ|வைத்திய|டெங்கு|கொவிட்|வைரஸ்|நோய்|நோயாளர்|தடுப்பூசி|"
              "புற்றுநோய்|தாதியர்|சத்திரசிகிச்சை|அறுவை சிகிச்சை|மருந்து|சிக்குன்குனியா|காய்ச்சல்|சிறுநீரக|"
              "போஷாக்கு|உலக சுகாதார",
    "sports": "கிரிக்கெட்|போட்டி|ஒருநாள்|டி20|டெஸ்ட்|உலகக் கிண்ண|விக்கெட்|இன்னிங்ஸ்|துடுப்பாட்ட|"
              "பந்துவீச்ச|கால்பந்து|உதைபந்தாட்ட|றக்பி|ரக்பி|மெய்வல்லுநர்|ஒலிம்பிக்|ஆசியக் கிண்ண|விளையாட்டு|"
              "அணித்தலைவர்|பதக்க|கரப்பந்தாட்ட|வலைப்பந்தாட்ட|சம்பியன்",
    "education": "பாடசாலை|மாணவர்|மாணவி|ஆசிரியர்|பல்கலைக்கழக|பரீட்சை|உயர்தர|சாதாரண தர|புலமைப்பரிசில்|"
                 "கல்வி|பட்டதாரி|பட்டப்படிப்பு|அதிபர்|தனியார் வகுப்பு|விரிவுரையாளர்|கல்விச் சீர்திருத்த",
    "international": "இஸ்ரேல்|காசா|ஹமாஸ்|ஈரான்|உக்ரைன்|ரஷ்ய|புட்டின்|புடின்|ட்ரம்ப்|டிரம்ப்|பைடன்|அமெரிக்க|"
                     "பாகிஸ்தான்|பங்களாதேஷ்|நேபாள|மாலைதீவு|மியன்மார்|ஆப்கானிஸ்தான்|சீன|லெபனான்|சிரியா|"
                     "ஐரோப்பிய|மோடி|டெல்லி|தமிழ்நாடு|தமிழக|ஜப்பான்|கொரிய|தாய்வான்|சவூதி|பிரித்தானிய|"
                     "பிரான்ஸ்|ஜேர்மனி|கனடா|அவுஸ்திரேலிய|இந்திய",
    "other": "திரைப்பட|சினிமா|நடிகர்|நடிகை|பாடகர்|பாடல்|இசை|இசைநிகழ்ச்சி|வெசாக்|பொசன்|பெரஹரா|"
             "ஆலய|கோவில்|தேவாலய|பள்ளிவாசல்|மத வழிபாட்டு|தேரர்|ஆயர்|பாப்பரசர்|திருவிழா|உற்சவ|புத்தாண்டு|"
             "நத்தார்|றமழான்|ரமழான்|தீபாவளி|பொங்கல்|கலைஞர்|கண்காட்சி|கலாசார|தொழில்நுட்ப|செயற்கை நுண்ணறிவு|"
             "கையடக்கத் தொலைபேசி|வட்ஸ்அப்|பேஸ்புக்|யானை|வனஜீவராசி|சிறுத்தை|சுற்றாடல்",
}

SRI_LANKA = re.compile(r"sri lanka|lanka|colombo|ශ්‍රී ලංකා|ලංකා|කොළඹ|இலங்கை|கொழும்பு", re.I)


def _compile():
    rules = {}
    for lang, table in (("en", EN), ("si", SI), ("ta", TA)):
        rules[lang] = {}
        for topic, words in table.items():
            alts = sorted({w.strip() for w in words.split("|") if w.strip()}, key=len, reverse=True)
            body = "|".join(re.escape(a) for a in alts)
            # English: whole words. Sinhala/Tamil: stems, so substring match.
            pat = rf"(?<![a-z0-9])(?:{body})(?![a-z0-9])" if lang == "en" else rf"(?:{body})"
            rules[lang][topic] = re.compile(pat, re.I)
    return rules


RULES = _compile()


def keyword_scores(title: str, lead: str, lang: str) -> dict[str, int]:
    r = RULES.get(lang, RULES["en"])
    title, lead = title.lower(), lead.lower()
    scores = {}
    for topic, pat in r.items():
        s = 2 * len(pat.findall(title)) + len(pat.findall(lead))
        if s:
            scores[topic] = s
    # "international" only when the story is not about Sri Lanka
    if "international" in scores and SRI_LANKA.search(title):
        scores.pop("international")
    return scores


def weak_label(newspaper_id: str, section: str, title: str, body: str, lang: str):
    """Return (label, source) or (None, None)."""
    sec = SECTION_LABELS.get((newspaper_id, section))
    if sec:
        return sec, "section"
    scores = keyword_scores(title, body[:300], lang)
    if not scores:
        return None, None
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0
    if s1 >= 3 and s1 - s2 >= 2:
        return best, "keywords"
    return None, None
