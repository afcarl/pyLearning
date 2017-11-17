import pandas as pd
import numpy as np
import string
import operator
import spacy

df = pd.read_csv("1hb_2014.csv")
new_york_city_names = [
    "NEW YORK,", "BAYSIDE", "1 COURT SQUARE",
    "160 FORT WASHINGTON AVENUE", "ASTORIA",
    "LONG IS CITY", "RIDGEWOOD", "BROOKYLN",
    "160 CONVENT AVENUE", "227 EAST 56TH STREET",
    "SOUTH OZONE PARK", "BROOKLY", "LONG ISLAND CTY",
    "DOWNTOWN BROOKLYN", "BRONX,NEW YORK", "5 TIMES SQUARE",
    "MANHATTAN", "MANHATAN", "BROOKLYN,", "BROOKLN",
    "MEW YORK", "BRONX", "132 HARRISON PLACE",
    "LONGISLAND CITY", "NEW YORK", "MIDTOWN MANHATTAN",
    "METRO NEW YORK", "NEW YOUR", "BROOKYN",
    "NEW YORK, 10003", "BROOKLYN", "NEW TORK",
    "650 WEST 168TH STREET", "325 HUDSON STREET, 9TH FLOOR",
    "FORREST HILLS", "BRONX.NEW YORK", "CITY OF NEW YORK",
    "JACKSON HEIGHTS", "FLUSHING", "THROGGS NECK",
    "QUEENS VILLAGE", "NEW ORK", "JOHNSON STREET",
    "ROCKAWAY PARK", "LONG ISLAND CITY, QUEENS",
    "NEW YORK,NEW YORK", "MANHATTAN BEACH",
    "NEW YORK, NEW YORK", "NEW YORK CITY",
    "NEWYORK CITY", "S RICHMOND HILL",
    "L.I.C.", "RICHMOND HILL", "55 WATER STREET",
    "OZONE PARK", "LONG ISLANDY CITY", "JAMAICA",
    "INWOOD", "SUNNYSIDE", "NEW YORK NY",
    "CAMBRIA HEIGHTS", "NEW YORK CITY,",
    "THROGS NECK", "137 VARICK STREET, 2ND FLOOR",
    "#35620 OZONE PARK NY MSA", "JAMAICA, NY",
    "LONG ISLAND CITY,", "NEW YOURK", "NY",
    "NEWYORK", "YORKERS", "NEW YROK",
    "LIC", "13 W.100TH STREET, APT. 4B",
    "850 12TH AVENUE", "NEW YOK",
    "701 WEST 168TH STREET", "FLUSHING, NY",
    "630 WEST 168TH STREET", "FOREST HILLS",
    "BOWLING GREEN", "NEW YORK, NY -", "LITTLE NECK",
    "NEW YORK, NY", "43-34 32ND PLACE",
    "LONG ISLAD CITY", "REGO PARK",
    "MANHATTAN, NEW YORK", "BRONX,NY",
    "LONG ISLAND CITY", "NYC",
    "QUEENS", "YONKERS", "BRONX, NEW YORK"
]

def segment_to_nyc(x):
    if x in new_york_city_names:
        return True
    else:
        return False

def clean_split(employer):
    translations = [employer.maketrans(elem," ") for elem in string.punctuation]
    stop_words = ["INC", "LLC", "&", "PC", "LP", "LLP", "CO", "LTD", "CORP"]
    for translation in translations:
        employer = employer.translate(translation)
    employer = ''.join(employer)
    return [elem for elem in employer.split() if elem not in stop_words]
    
nlp = spacy.load('en')
nyc = df[df["lca_case_workloc1_city"].apply(segment_to_nyc)]
employers = list(nyc["lca_case_employer_name"].unique())
employer_counts = {}
for index, employer in enumerate(employers):
    other_employers = employers[:index] + employers[index+1:]
    names = clean_split(employer)
    names = [name for name in names if len(name) > 1]
    names = [name for name in names if not name.isdigit()]
    doc = nlp(' '.join(names))
    names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    tmp = {}.fromkeys(names, 0)
    for name in names:
        tmp[name] += sum([employer.count(name) for employer in other_employers])
    employer_counts.update(tmp)


print(max(employer_counts.items(), key=operator.itemgetter(1)))

counter = 0
for employer in employer_counts:
    if employer_counts[employer] < 100:
        counter += 1
