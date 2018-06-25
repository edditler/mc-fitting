from aiida.orm.data.structure import StructureData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.base import Int, Str, Bool
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.code import Code

from aiida.work.workchain import WorkChain, ToContext, Calc, while_
from aiida.work.run import submit

from aiida_cp2k.calculations import Cp2kCalculation

import tempfile
import shutil


class EnergyWorkChain(WorkChain):

    @classmethod
    def define(cls, spec):
        super(EnergyWorkChain, cls).define(spec)
        spec.input("cp2k_code", valid_type=Code)
        spec.input("structure", valid_type=StructureData)
        spec.input("num_machines", valid_type=Int, default=Int(1))
        spec.input("wavefunction", valid_type=Str, default=Str(''))
        spec.input("gasphase", valid_type=Bool, default=Bool(False))

        spec.outline(
            cls.run_ene,
            while_(cls.not_converged)(
                cls.run_ene_again
            ),
        )
        spec.dynamic_output()

    # ==========================================================================
    def not_converged(self):
        return self.ctx.geo_opt.res.exceeded_walltime

    # ==========================================================================
    def run_ene(self):
        self.report("Running CP2K geometry optimization")

        inputs = self.build_calc_inputs(self.inputs.structure,
                                        self.inputs.cp2k_code,
                                        self.inputs.num_machines,
                                        self.inputs.wavefunction,
                                        None,
                                        self.inputs.gasphase)

        self.report("inputs: "+str(inputs))
        future = submit(Cp2kCalculation.process(), **inputs)
        return ToContext(geo_opt=Calc(future))

    # ==========================================================================
    def run_ene_again(self):
        # TODO: make this nicer.
        inputs_new = self.build_calc_inputs(self.inputs.structure,
                                            self.inputs.cp2k_code,
                                            self.inputs.num_machines,
                                            self.inputs.wavefunction,
                                            self.ctx.geo_opt.out.remote_folder,
                                            self.inputs.gasphase)

        self.report("inputs (restart): "+str(inputs_new))
        future_new = submit(Cp2kCalculation.process(), **inputs_new)
        return ToContext(geo_opt=Calc(future_new))

    # ==========================================================================
    @classmethod
    def build_calc_inputs(cls, structure, code, num_machines,
                          wavefunction, remote_calc_folder, gasphase):

        inputs = {}
        if gasphase:
            inputs['_label'] = "ft_ene_gas"
        else:
            inputs['_label'] = "ft_ene"
        inputs['code'] = code
        inputs['file'] = {}

        # write the xyz structure file
        tmpdir = tempfile.mkdtemp()

        atoms = structure.get_ase()  # slow
        slab = atoms[-1568:]
        mol = atoms[:-1568]

        molslab_fn = tmpdir + '/mol_on_slab.xyz'
        if gasphase:
            mol.write(molslab_fn)
        else:
            atoms.write(molslab_fn)

        molslab_f = SinglefileData(file=molslab_fn)
        inputs['file']['molslab_coords'] = molslab_f

        shutil.rmtree(tmpdir)

        # parameters
        cell_abc = "%f  %f  %f" % (atoms.cell[0, 0],
                                   atoms.cell[1, 1],
                                   atoms.cell[2, 2])

        walltime = 86000

        inp = cls.get_cp2k_input(cell_abc,
                                 wavefunction,
                                 walltime*0.97)

        if remote_calc_folder is not None:
            inp['EXT_RESTART'] = {
                'RESTART_FILE_NAME': './parent_calc/aiida-1.restart'
            }
            inputs['parent_folder'] = remote_calc_folder

        inputs['parameters'] = ParameterData(dict=inp)

        # settings
        settings = ParameterData(dict={'additional_retrieve_list': ['*.xyz']})
        inputs['settings'] = settings

        # resources
        inputs['_options'] = {
            "resources": {"num_machines": num_machines},
            "max_wallclock_seconds": walltime,
        }

        return inputs

    # ==========================================================================
    @classmethod
    def get_cp2k_input(cls, cell_abc, wavefunction, walltime):

        inp = {
            'GLOBAL': {
                'RUN_TYPE': 'ENERGY_FORCE',
                'WALLTIME': '%d' % walltime,
                'PRINT_LEVEL': 'LOW'
            },
            'FORCE_EVAL': cls.get_force_eval_qs_dft(cell_abc, wavefunction),
        }
        return inp

    # ==========================================================================
    @classmethod
    def get_force_eval_qs_dft(cls, cell_abc, wavefunction):
        force_eval = {
            'METHOD': 'Quickstep',
            'DFT': {
                'BASIS_SET_FILE_NAME': 'BASIS_MOLOPT',
                'POTENTIAL_FILE_NAME': 'POTENTIAL',
                'RESTART_FILE_NAME': wavefunction,
                'QS': {
                    'METHOD': 'GPW',
                    'EXTRAPOLATION': 'ASPC',
                    'EXTRAPOLATION_ORDER': '3',
                    'EPS_DEFAULT': '1.0E-14',
                },
                'MGRID': {
                    'CUTOFF': '600',
                    'NGRIDS': '5',
                },
                'SCF': {
                    'MAX_SCF': '20',
                    'SCF_GUESS': 'RESTART',
                    'EPS_SCF': '1.0E-7',
                    'OT': {
                        'PRECONDITIONER': 'FULL_SINGLE_INVERSE',
                        'MINIMIZER': 'CG',
                    },
                    'OUTER_SCF': {
                        'MAX_SCF': '15',
                        'EPS_SCF': '1.0E-7',
                    },
                    'PRINT': {
                        'RESTART': {
                            'EACH': {
                                'QS_SCF': '0'
                            },
                            'ADD_LAST': 'NUMERIC',
                            'FILENAME': 'RESTART'
                        },
                        'RESTART_HISTORY': {'_': 'OFF'}
                    }
                },
                'XC': {
                    'XC_FUNCTIONAL': {'_': 'PBE'},
                    'VDW_POTENTIAL': {
                        'DISPERSION_FUNCTIONAL': 'PAIR_POTENTIAL',
                        'PAIR_POTENTIAL': {
                            'TYPE': 'DFTD3',
                            'CALCULATE_C9_TERM': '.TRUE.',
                            'PARAMETER_FILE_NAME': 'dftd3.dat',
                            'REFERENCE_FUNCTIONAL': 'PBE',
                            'R_CUTOFF': '15',
                        }
                    }
                },
            },
            'SUBSYS': {
                'CELL': {'ABC': cell_abc},
                'TOPOLOGY': {
                    'COORD_FILE_NAME': 'mol_on_slab.xyz',
                    'COORDINATE': 'xyz',
                },
                'KIND': [],
            },
            'PRINT': {
                'FORCES': {
                    '_': 'ON',
                    'ADD_LAST': 'NUMERIC',
                    'FILENAME': 'forces'
                }
            }
        }

        force_eval['SUBSYS']['KIND'].append({
            '_': 'Au',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q11'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'C',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q4'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'Br',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'O',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q6'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'N',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q5'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'I',
            'BASIS_SET': 'DZVP-MOLOPT-SR-GTH',
            'POTENTIAL': 'GTH-PBE-q7'
        })
        force_eval['SUBSYS']['KIND'].append({
            '_': 'H',
            'BASIS_SET': 'TZV2P-MOLOPT-GTH',
            'POTENTIAL': 'GTH-PBE-q1'
        })

        return force_eval

# EOF
