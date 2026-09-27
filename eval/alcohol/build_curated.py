"""Write independently authored challenge cases for the synthetic benchmark."""

from pathlib import Path

from alcohol_ft.data import write_jsonl


# Each entry is (scenario, notes). A vertical bar separates encounters in one visit.
CASES = {
    "direct": [
        ("current_use", "Triage for wrist pain. | Physician: patient had four beers before slipping on the steps."),
        ("current_use", "Abdominal symptoms reviewed. | Social history: wine with dinner on most nights."),
        ("current_use", "Patient seen after a fall. | He says he drank whiskey at a party before the event."),
        ("current_use", "Headache workup completed. | She reports one cocktail every Friday."),
        ("current_use", "Nurse records no drinking at triage. | Later physician interview: patient admits two glasses of wine today."),
        ("current_use", "Medication reconciliation done. | Patient describes binge drinking twice monthly."),
        ("current_use", "ED note: odor of alcohol and patient states he consumed vodka this morning."),
        ("current_use", "The patient tells the consultant she drinks cider several evenings a week."),
        ("current_use", "Routine surgical consult. | Patient describes a beer after work each day."),
        ("current_use", "Patient says no alcohol today. | History from the same patient: drinks heavily on weekends."),
        ("past_use", "Discharge note: in recovery from alcohol dependence, sober for eight years."),
        ("past_use", "The patient formerly drank a bottle of spirits daily; stopped after rehabilitation."),
        ("past_use", "Social history notes past hazardous drinking but current abstinence."),
        ("past_use", "Patient reports prior alcohol misuse in college; no recent consumption."),
        ("past_use", "Old drinking history reviewed: patient quit alcohol five years ago."),
        ("condition", "Gastroenterology assesses chronic pancreatitis secondary to this patient's alcohol use."),
        ("condition", "Hospitalist records alcohol-related cirrhosis from years of patient drinking."),
        ("condition", "Patient admitted for withdrawal after abruptly stopping daily alcohol consumption."),
        ("condition", "Diagnosis: alcoholic ketoacidosis associated with patient's recent heavy drinking."),
        ("condition", "Neurology attributes the patient's neuropathy to long-term alcohol use."),
        ("condition", "Treatment team documents alcohol use disorder in remission."),
        ("condition", "Serum ethanol is elevated; patient confirms drinking before arrival."),
        ("multi_encounter", "Triage records a negative alcohol screen. | Consultant later documents that patient drinks six beers nightly."),
        ("multi_encounter", "ED physician: patient was struck by a drunk driver. | Ward note: patient also had two drinks before the collision."),
        ("multi_encounter", "Nurse: patient denies alcohol today. | Discharge summary: remote history of alcohol dependence."),
        ("multi_encounter", "Initial note: social history unavailable. | Follow-up note: patient reports regular spirits consumption."),
        ("multi_encounter", "Trauma note: partner had been drinking. | Patient separately acknowledges drinking beer that evening."),
        ("abbreviation", "Social hx: EtOH 2 drinks/day per pt."),
        ("abbreviation", "Patient reports ETOH use only on holidays."),
        ("abbreviation", "PMH includes AUD, now in sustained remission."),
    ],
    "indirect": [
        ("impaired_driver", "Sober passenger with rib fractures after collision caused by a driver who had been drinking."),
        ("impaired_driver", "Patient was walking through a crossing when an intoxicated motorist hit her; she denies alcohol use."),
        ("impaired_driver", "Police report driver of other vehicle failed alcohol test. Patient was a nondrinking cyclist."),
        ("impaired_driver", "Child was injured in a crash; parent driver had consumed alcohol. Child is the patient."),
        ("impaired_driver", "An impaired bus driver caused the collision that brought this abstinent patient to the ED."),
        ("impaired_driver", "Trauma team: patient was asleep in parked car struck by drunk motorist. No patient drinking."),
        ("assault", "Patient sustained facial injury from an intoxicated assailant and reports no personal drinking."),
        ("assault", "An inebriated spouse pushed the patient; the patient has never used alcohol."),
        ("assault", "Sober patient was hit with a bottle by a person who had been drinking."),
        ("assault", "Injuries followed violence by a drunk roommate. Patient denies alcohol use."),
        ("assault", "The patient's intoxicated caregiver dropped the patient, causing the injury."),
        ("assault", "ED note: intoxicated stranger shoved the patient into traffic; patient was sober."),
        ("prenatal", "Infant evaluated for suspected fetal alcohol spectrum disorder following birth mother's prenatal drinking."),
        ("prenatal", "Child's developmental findings are linked to alcohol exposure in utero; mother drank during pregnancy."),
        ("prenatal", "Neonatal note: maternal alcohol intake during gestation caused fetal exposure; newborn is patient."),
        ("prenatal", "Pediatric service records prenatal alcohol exposure from biological mother's use."),
        ("prenatal", "The patient is a child with alcohol-related birth defects due to maternal consumption."),
        ("prenatal", "Infant's withdrawal-like symptoms are attributed to prenatal exposure to the mother's drinking."),
        ("other_person", "Patient burned when an intoxicated neighbor left a stove on; patient reports abstinence."),
        ("other_person", "Sober patient injured while carrying an alcohol-impaired friend downstairs."),
        ("other_person", "Patient fell fleeing a driver who was drunk and entered the sidewalk."),
        ("other_person", "An intoxicated coworker mishandled machinery, injuring the patient who did not drink."),
        ("other_person", "Patient was knocked over by a person who had been consuming alcohol; no patient use documented."),
        ("other_person", "Patient injured after their intoxicated parent dropped a heavy object on them."),
        ("multi_encounter", "Triage: ankle fracture, denies EtOH. | Physician: fracture occurred when a drunk driver struck patient."),
        ("multi_encounter", "First note: child with developmental delay. | Specialist: condition stems from maternal alcohol use in pregnancy."),
        ("multi_encounter", "Nursing note: no alcohol use by patient. | Trauma note: intoxicated partner caused fall."),
        ("multi_encounter", "ED note: passenger injured in crash. | Police history reviewed: driver had been drinking; patient did not."),
        ("multi_encounter", "Admission note: sober patient with smoke inhalation. | Fire started by intoxicated neighbor."),
        ("multi_encounter", "Pediatric note: bruising after fall. | Caregiver was impaired by alcohol and dropped the child."),
    ],
    "negative": [
        ("denial", "Patient denies ever drinking alcohol. Visit for knee swelling."),
        ("denial", "Social history: no EtOH. Evaluation is for viral illness."),
        ("denial", "Patient says she does not consume alcoholic beverages; headache treated."),
        ("denial", "Routine questionnaire asks about drinking; answer marked never."),
        ("denial", "Patient reports complete abstinence and no prior alcohol use."),
        ("denial", "Nursing screening: alcohol consumption zero in the past year or before."),
        ("screen_or_lab", "Alcohol screen ordered as standard preoperative assessment; result negative."),
        ("screen_or_lab", "Ethanol level below assay detection; no use history or external cause recorded."),
        ("screen_or_lab", "Provider asks about drinking but patient does not answer; no other alcohol evidence."),
        ("screen_or_lab", "Routine AUDIT-C form left blank. Visit for cough."),
        ("screen_or_lab", "Toxicology panel includes ethanol, reported negative."),
        ("screen_or_lab", "Alcohol use field reads unknown; no attribution in the notes."),
        ("counseling", "Discharge paperwork advises avoiding alcohol with metronidazole; no consumption documented."),
        ("counseling", "Medication instructions warn that alcohol can worsen drowsiness. Patient's use is not recorded."),
        ("counseling", "Preventive leaflet discusses limits on drinking; this patient's visit is for dermatitis."),
        ("counseling", "Clinician offers general advice about alcohol during pregnancy, without documenting exposure."),
        ("counseling", "Patient asks if wine is safe with a new drug; no statement that wine was consumed."),
        ("counseling", "Chart automatically includes an alcohol cessation handout; no use noted."),
        ("family", "Father has alcohol dependence. Patient denies drinking and presents with unrelated sprain."),
        ("family", "Family history lists an aunt's heavy drinking; not related to patient's asthma visit."),
        ("family", "Sibling is in alcohol rehabilitation. Patient's infection has no documented alcohol cause."),
        ("family", "Parent used alcohol years after patient's birth. No patient exposure or visit connection."),
        ("family", "Patient worries about partner's drinking, but this visit concerns an unrelated rash."),
        ("family", "Relative smelled of alcohol in waiting room; patient is here for kidney stone."),
        ("product", "Skin prepared with alcohol wipes for IV placement."),
        ("product", "Specimen container cleaned with isopropyl alcohol."),
        ("product", "Alcohol-based hand sanitizer used before examination."),
        ("product", "Prescription contains benzyl alcohol as an excipient; no consumption documented."),
        ("multi_encounter", "Triage: ankle pain, no social history. | Physician: alcohol screen negative; mechanical fall."),
        ("multi_encounter", "ED note: patient asks about EtOH interactions. | Discharge note: no drinking history documented."),
    ],
}


def main():
    rows = []
    for label, cases in CASES.items():
        assert len(cases) == 30, (label, len(cases))
        for index, (scenario, notes) in enumerate(cases, 1):
            case_id = f"curated-{label}-{index:02d}"
            rows.append({"case_id": case_id, "patient_id": case_id, "label": label,
                         "source": "curated_synthetic", "scenario": scenario,
                         "encounters": [{"type": "clinical note", "note": note.strip()}
                                        for note in notes.split("|")]})
    write_jsonl(Path("eval/alcohol/curated.jsonl"), rows)
    print(f"wrote {len(rows)} curated synthetic cases")


if __name__ == "__main__":
    main()
