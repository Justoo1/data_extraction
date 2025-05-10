from django.core.management.base import BaseCommand
from extractor.models import SENDIGDomain
import json

class Command(BaseCommand):
    help = 'Load SENDIG domain definitions'

    def handle(self, *args, **options):
        # Define SENDIG domains with their required variables
        domains = [
            {
                'code': 'DM',
                'name': 'Demographics',
                'description': 'Contains demographic data such as species, sex, and age.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'RFSTDTC', 'RFENDTC', 
                    'RFXSTDTC', 'RFXENDTC', 'SEX', 'SPECIES', 'STRAIN', 'AGE', 'AGEU'
                ])
            },
            {
                'code': 'TX',
                'name': 'Trial Sets',
                'description': 'Contains sets of planned treatments as defined in the protocol.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'SETCD', 'SET', 'TXVAL'
                ])
            },
            {
                'code': 'TS',
                'name': 'Trial Summary',
                'description': 'Contains trial-level data such as study design, species, and test article.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'TSSEQ', 'TSGRPID', 'TSPARMCD', 'TSPARM', 'TSVAL'
                ])
            },
            {
                'code': 'TE',
                'name': 'Trial Elements',
                'description': 'Contains planned elements of the study design.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'ETCD', 'ELEMENT', 'TESTRL', 'TEENRL'
                ])
            },
            {
                'code': 'TA',
                'name': 'Trial Arms',
                'description': 'Contains trial arms (groups) information.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'ARMCD', 'ARM', 'TAETORD', 'ETCD', 'ELEMENT'
                ])
            },
            {
                'code': 'LB',
                'name': 'Laboratory Test Results',
                'description': 'Contains laboratory test results including hematology, clinical chemistry, and urinalysis.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'LBSEQ', 'LBTESTCD', 'LBTEST', 
                    'LBCAT', 'LBORRES', 'LBORRESU', 'LBSPEC', 'LBNAM', 'LBDTC'
                ])
            },
            {
                'code': 'CL',
                'name': 'Clinical Observations',
                'description': 'Contains clinical observations made during the study.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'CLSEQ', 'CLTYPE', 'CLTEST', 
                    'CLORRES', 'CLDTC'
                ])
            },
            {
                'code': 'BW',
                'name': 'Body Weight',
                'description': 'Contains body weight measurements.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'BWSEQ', 'BWTESTCD', 'BWTEST', 
                    'BWORRES', 'BWORRESU', 'BWDTC'
                ])
            },
            {
                'code': 'FW',
                'name': 'Food Consumption',
                'description': 'Contains food consumption data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'FWSEQ', 'FWTESTCD', 'FWTEST', 
                    'FWORRES', 'FWORRESU', 'FWDTC'
                ])
            },
            {
                'code': 'MA',
                'name': 'Macroscopic Findings',
                'description': 'Contains macroscopic (gross) pathology findings.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'MASEQ', 'MAGRPID', 'MATESTCD',
                    'MATEST', 'MALOC', 'MAPORRES', 'MADTC'
                ])
            },
            {
                'code': 'MI',
                'name': 'Microscopic Findings',
                'description': 'Contains microscopic pathology findings.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'MISEQ', 'MIGRPID', 'MITESTCD',
                    'MITEST', 'MILOC', 'MISPEC', 'MIPORRES', 'MISEV', 'MIDTC'
                ])
            },
            {
                'code': 'EX',
                'name': 'Exposure',
                'description': 'Contains exposure (dosing) data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'EXSEQ', 'EXTRT', 'EXDOSE', 
                    'EXDOSU', 'EXROUTE', 'EXSTDTC', 'EXENDTC'
                ])
            },
            {
                'code': 'DS',
                'name': 'Disposition',
                'description': 'Contains disposition (completion status) data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'DSSEQ', 'DSDECOD', 'DSSTDTC'
                ])
            },
            {
                'code': 'PC',
                'name': 'Pharmacokinetic Concentrations',
                'description': 'Contains pharmacokinetic concentration data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'PCSEQ', 'PCTESTCD', 'PCTEST',
                    'PCORRES', 'PCORRESU', 'PCSPEC', 'PCDTC'
                ])
            },
            {
                'code': 'PP',
                'name': 'Pharmacokinetic Parameters',
                'description': 'Contains pharmacokinetic parameter data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'PPSEQ', 'PPTESTCD', 'PPTEST',
                    'PPORRES', 'PPORRESU'
                ])
            },
            {
                'code': 'OM',
                'name': 'Organ Measurements',
                'description': 'Contains organ measurement data.',
                'required_variables': json.dumps([
                    'STUDYID', 'DOMAIN', 'USUBJID', 'OMSEQ', 'OMTESTCD', 'OMTEST',
                    'OMORRES', 'OMORRESU', 'OMLOC', 'OMDTC'
                ])
            }
        ]
        
        # Create or update domains
        for domain_data in domains:
            SENDIGDomain.objects.update_or_create(
                code=domain_data['code'],
                defaults={
                    'name': domain_data['name'],
                    'description': domain_data['description'],
                    'required_variables': domain_data['required_variables']
                }
            )
            
        self.stdout.write(self.style.SUCCESS(f'Successfully loaded {len(domains)} SENDIG domains'))