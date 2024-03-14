from tdc.utils import retrieve_label_name_list
from tdc.single_pred import CRISPROutcome



label_list = retrieve_label_name_list('Leenay')
dataset = CRISPROutcome(name = 'Leenay', label_name = label_list[0])
data = dataset.get_data()
print(data)
print(data.columns)

def make_cripsr_dataset(
	path:str, 
	)