import pywapor
import os
import re
import requests
import numpy as np
import glob
import warnings
import json
import importlib
from osgeo import gdal
from functools import partial
from pywapor.general.logger import log, adjust_logger
from pywapor.collect.downloader import collect_sources
from pywapor.general.compositer import time_bins
from pywapor.general.processing_functions import adjust_timelim_dtype, func_from_string, open_ds, remove_ds, is_corrupt_or_empty, has_wrong_bb_or_period

class Configuration():

    variable_categories = {
        "optical": ["ndvi", "r0", "mndwi", "psri", "vari_red_edge", "bsi", "nmdi", "green", "nir"],
        "thermal": ["bt", "lst"],
        "solar radiation": ["ra_flat"],
        "precipitation": ["p"],
        "elevation": ["z", "slope", "aspect"],
        "meteorological": ["t_air", "t_air_min", "t_air_max", "u", "vp", "p_air", "p_air_0", "u2m", "v2m", "qv", "wv", "t_dew"],
        "statics": ["lw_slope", "lw_offset", "z_obst_max", "rs_min", "land_mask", "vpd_slope", "t_opt", "t_amp", "t_amp_year", "rn_slope", "rn_offset", "z_oro"],
        "soil moisture": ["se_root"],
    }

    category_variables = {}
    for x in [{var: cat for var in varis} for cat, varis in variable_categories.items()]:
        category_variables = {**category_variables, **x}

    se_root_vars = [
        [("ndvi",)], 
        [("bt", "lst"), ("lst",), ("bt",)], 
        [("u",), ("u2m", "v2m")],
        [("p_air",)], 
        [("p_air_0",)], 
        [("t_air",)], 
        [("wv",)], 
        [("t_dew",), ("qv",)]
        ]
    
    et_look_vars = [
        [("ndvi",)],
        [("r0",)],

        [("se_root",)], 

        [("p",)],
        [("z", "slope", "aspect"), ("z")],
        [("ra_flat",)],

        [("u",), ("u2m", "v2m")],
        [("p_air",)], 
        [("p_air_0",)], 
        [("t_air",)],
        [("t_air_min",)], 
        [("t_air_max",)], 
        [("t_dew",), ("qv",)],

        [("rn_offset",)],
        [("rn_slope",)],
        [("t_amp",),],
        [("t_opt",)],
        [("vpd_slope",)],
        [("lw_offset",)],
        [("lw_slope",)],

        [("rs_min",), ()],
        [("land_mask",), ()],
        [("z_obst_max",), ()],
        [("z_oro",), ()],
        ]
    
    @staticmethod
    def source_func(x):
        return x if "FILE:" in x else x.split(".")[0]
    
    @staticmethod
    def pname_func(x):
        return "none" if "FILE:" in x else ".".join(x.split(".")[1:])

    def __init__(self, full = None, summary = None, se_root = None, 
                 et_look = None):
        self.full = full
        self.summary = summary
        self.se_root = se_root
        self.et_look = et_look

    def __repr__(self):
        summary = self.summary.copy()

        example_product = summary.pop('_EXAMPLE_')
        whittaker = summary.pop('_WHITTAKER_')
        sharpen = summary.pop('_ENHANCE_')

        base = "\n".join([f"--> {cat.title()} data:\n    > `{'`, `'.join(sources)}`" for cat, sources in summary.items()])

        additional = f"""
--> Example Product:
    > {example_product}
--> Applying whittaker to variables from sources:
    > `{'` and `'.join(whittaker)}`
--> Sharpening these variables:
    > `{'` and `'.join(sharpen)}`"""
        return base + additional

    @staticmethod
    def default_enhancers(prod, var):
        source = Configuration.source_func(prod)
        if "FILE:" in source:
            return []
        product_name = Configuration.pname_func(prod)
        mod = importlib.import_module(f"pywapor.collect.product.{source}")
        x = mod.default_post_processors(product_name, [var])[var]

        funcs = list()
        for x_ in x:
            if isinstance(x_, partial):
                funcs.append({
                    "func": f"{x_.func.__module__}.{x_.func.__name__}",
                    "args": x_.args,
                    "keywords": x_.keywords,
                })
            else:
                funcs.append({"func": f"{x_.__module__}.{x_.__name__}"})
        
        return funcs

    @classmethod
    def from_name(cls, name):
        log.info(f"--> Searching configuration for `{name}`.").add()

        folder = os.path.realpath(pywapor.__path__[0])
        available_configs = glob.glob(os.path.join(folder, "configs", "*.json"))
        available_names = sorted([os.path.splitext(os.path.split(x)[-1])[0] for x in available_configs])
        log.info(f"> Available configurations are `{'`, `'.join(available_names)}`")
    
        synonyms = {
            "level_1": "WaPOR2_level_1",
            "level_2": "WaPOR2_level_2",
            "level_3": "WaPOR2_level_3",
            "level_1_v2": "WaPOR2_level_1",
            "level_2_v2": "WaPOR2_level_2",
            "level_3_v2": "WaPOR2_level_3",
            "level_1_v3": "WaPOR3_level_1",
            "level_2_v3": "WaPOR3_level_2",
            "level_3_v3": "WaPOR3_level_3",
        }

        name_ = synonyms.get(name, name)

        selection = None
        for fh in available_configs:
            fn = os.path.splitext(os.path.split(fh)[-1])[0]
            if fn == name_:
                selection = fh
                break

        if isinstance(selection, type(None)):
            raise ValueError(f"No configuration found for name `{name}`, choose one from `{'`, `'.join(available_names)}`.")
        else:
            config = Configuration.from_json(selection)

        log.sub().info("--> Configuration set.")

        return config
    
    def to_json(self, fh):
        out = {
            "summary": self.summary, 
            "full": self.full, 
            "se_root": self.se_root, 
            "et_look": self.et_look,
        }
        class EncodeSetTuple(json.JSONEncoder):
            def encode(self, obj):
                def hint_tuples(item):
                    if isinstance(item, tuple):
                        return {'type': "tuple", 'value': list(item)}
                    if isinstance(item, list):
                        return [hint_tuples(e) for e in item]
                    if isinstance(item, set):
                        return {'type': "set", 'value': list(item)}
                    if isinstance(item, dict):
                        return {key: hint_tuples(value) for key, value in item.items()}
                    else:
                        return item
                return super(EncodeSetTuple, self).encode(hint_tuples(obj))
        
        json_string = json.dumps(out, cls = EncodeSetTuple)
        with open(fh, "w", encoding='utf8') as x:
            x.write(json_string)

    @classmethod
    def from_json(cls, fh):
        log.info(f"--> Loading configuration from `{os.path.split(fh)[1]}`.").add()
        def decode_set(dct):
            if dct.get("type", None) == "set":
                return set(dct.get('value'))
            if dct.get("type", None) == "tuple":
                return tuple(dct.get('value'))
            return dct
        with open(fh, "r", encoding='utf8') as fp:
            config_ = json.load(fp, object_hook=decode_set)
        
        config = cls(full = config_["full"], summary = config_["summary"], se_root = config_["se_root"], 
                 et_look = config_["et_look"])

        config.validate()
        log.sub().info("--> Configuration loaded.")
        return config
    
    @classmethod
    def from_summary(cls, summary):
        log.info("--> Creating configuration from summary.").add()

        valids = set(cls.variable_categories.keys()).union({"_EXAMPLE_", "_WHITTAKER_", "_ENHANCE_"})
        invalid_keys = set(summary.keys()).difference(valids)
        if invalid_keys:
            log.warning(f"--> Summary contains unrecognized keys ({invalid_keys}), they will be ignored.")
            summary = {k:v for k,v in summary.items() if k not in invalid_keys}

        example_product = summary.pop('_EXAMPLE_', '')
        temporal_interp = summary.pop('_WHITTAKER_', {})
        enhance = summary.pop('_ENHANCE_', {})

        full = dict()

        unique_enhancers = set([item for row in list(enhance.values()) for item in row])
        extra_vars = list()
        for enhancer in unique_enhancers:
            func = func_from_string(enhancer)
            extra_vars += [[(x,)] for x in func.__kwdefaults__["req_vars"]]

        for var_group in cls.et_look_vars + cls.se_root_vars + extra_vars:
            for var_ in var_group:
                for var in var_:

                    if var in full.keys():
                        continue

                    cat = cls.category_variables[var]
                    possible_prods = summary.get(cat, None)
                    
                    if isinstance(possible_prods, type(None)):
                        continue

                    products = []
                    t_interps = list()
                    for prod in possible_prods:
                        valid = cls.has_var(prod, var)
                        if valid:

                            enhancers = cls.default_enhancers(prod, var)

                            products += [{
                                "source": cls.source_func(prod), 
                                "product_name": cls.pname_func(prod), 
                                "enhancers": enhancers,
                                "is_example": prod == example_product,
                                }]
                            
                        x = temporal_interp.get(prod, None)
                        if not isinstance(x, type(None)):
                            t_interps += [x]
                    
                    if len(products) == 0:
                        continue

                    t_interps_unique = list(map(json.loads, set(map(lambda x: json.dumps(x, sort_keys=True), t_interps))))
                    # If no particular t_interp has been specified for this variable, use "linear".
                    if len(t_interps_unique) == 0:
                        temporal_interp_ = "linear"
                    # If a single unique t_interp has been specified, use that.
                    elif len(t_interps_unique) == 1:
                        temporal_interp_ = t_interps_unique[0]
                    # If multiple t_interp are specified for one variable, warn and use the first one.
                    else:
                        warn_string = str({"_WHITTAKER_": {x: temporal_interp[x] for x in possible_prods}})
                        log.warning(f"--> Multiple temporal interpolations specified for {var} (through `{warn_string}`), will continue using {t_interps_unique[0]}.")
                        temporal_interp_ = t_interps_unique[0]

                    variable_enhancers = enhance.get(var, [])

                    composite_type = "mean" if var.split("_")[-1] not in ["min", "max"] else var.split("_")[-1]

                    full[var] = {
                        "products": products,
                        "temporal_interp": temporal_interp_,
                        "variable_enhancers": variable_enhancers,
                        "spatial_interp": "bilinear",
                        "composite_type": composite_type,
                    }

        config = cls(full = full)
        config.validate()
        config.summarize()
        config.update_se_root_config()
        config.update_et_look_config()
        log.sub().info("--> Configuration created.")
        return config

    @staticmethod
    def has_var(prod, var, verbose = True):
        source = Configuration.source_func(prod)
        product_name = Configuration.pname_func(prod)
        if "FILE:" in source:
            if not verbose:
                log.info(f"> Variable `{var}` will be loaded from a file `{source}`.")
            return True
        mod = importlib.import_module(f"pywapor.collect.product.{source}")
        try:
            mod.default_vars(product_name, [var])
            valid = True
        except TypeError:
            valid = False
            if not verbose:
                log.warning(f"> {source}.{product_name} does not have a variable called `{var}`.") 
        return valid
    
    def validate(self):
        log.info("--> Validating configuration.").add()
        valids = []
        for var in self.full:
            for product in self.full[var]["products"]:
                prod = f"{product['source']}.{product['product_name']}"
                valid = Configuration.has_var(prod, var, verbose = False)
                valids.append(valid)
        if all(valids):
            log.info("> All specified variables sources are valid.")
        log.sub()
        return valid

    def update_se_root_config(self):
        log.info("--> Making configuration for SE_ROOT.").add()
        
        sharpened_vars = {var for var, config in self.full.items() if any(["sharpen" in x for x in config["variable_enhancers"]])}

        sharpeners = ["mndwi", "psri", "vari_red_edge", "bsi", "nmdi", "green", "nir"]
    
        se_root_config = dict()
        for var_group in self.se_root_vars:
            for var in var_group:
                if all([var_ in self.full.keys() for var_ in var]):
                    for var_ in var:
                        se_root_config[var_] = self.full[var_].copy()
                        _ = se_root_config[var_].pop("composite_type")
                    break
                elif var == var_group[-1]:
                    substring = [f"[`{'` and `'.join(x)}`]" for x in var_group]
                    log.warning(f"--> No configuration found for {' or '.join(substring)}.")
                else:
                    continue
        
        to_sharpen = set(sharpened_vars) & set(se_root_config)
        if to_sharpen:
            for var in sharpeners:
                if var in self.full.keys():
                    se_root_config[var] = self.full[var]
                else:
                    log.warning(f"--> No configuration found for `{var}` (required for sharpening of `{'` and `'.join(to_sharpen)}`).")

        self.se_root = se_root_config
        log.sub()

    def update_et_look_config(self):
        log.info("--> Making configuration for ET_LOOK.").add()
        
        sharpened_vars = {var for var, config in self.full.items() if any(["sharpen" in x for x in config["variable_enhancers"]])}

        sharpeners = ["mndwi", "psri", "vari_red_edge", "bsi", "nmdi", "green", "nir"]
    
        et_look_config = dict()
        for var_group in self.et_look_vars:
            for var in var_group:
                if all([var_ in self.full.keys() for var_ in var]):
                    for var_ in var:
                        et_look_config[var_] = self.full[var_].copy()
                    if len(var) == 0 and var == var_group[-1]:
                        substring = [f"[`{'` and `'.join(x)}`]" for x in var_group[:-1]]
                        log.info(f"--> No configuration found for {' or '.join(substring)}, will use constant value.")
                    break
                elif var == var_group[-1]:
                    substring = [f"[`{'` and `'.join(x)}`]" for x in var_group]
                    log.warning(f"--> No configuration found for {' or '.join(substring)}.")
                else:
                    continue
        
        to_sharpen = set(sharpened_vars) & set(et_look_config)
        if to_sharpen:
            for var in sharpeners:
                if var in self.full.keys():
                    et_look_config[var] = self.full[var]
                else:
                    log.warning(f"--> No configuration found for `{var}` (required for sharpening of `{'` and `'.join(to_sharpen)}`).")

        self.et_look = et_look_config
        log.sub()

    def summarize(self):
        log.info("--> Making summary of configuration.")
        
        summary = dict()
        for var, config in self.full.items():
            sources = set()
            for x in config["products"]:
                if "FILE:" in x["source"]:
                    sources.add(x["source"])
                else:
                    sources.add(f'{x["source"]}.{x["product_name"]}')
            cat = self.category_variables.get(var, "unknown")
            if cat == "other":
                log.warning(f"--> Unknown variable `{var}` found.")
            if cat in list(summary.keys()):
                summary[cat] = summary[cat].union(sources)
            else:
                summary[cat] = sources

        whittaker = dict()
        enhance = dict()
        example = None
        for k,v in self.full.items():
            if isinstance(example, type(None)):
                example_ = [f"{prod['source']}.{prod['product_name']}" for prod in v["products"] if prod["is_example"]]
                if len(example_) > 0:
                    example = example_[0]
            if isinstance(v["temporal_interp"], dict):
                if v["temporal_interp"].get("method", "") == "whittaker":
                    for prod in v["products"]:
                        whittaker[f"{prod['source']}.{prod['product_name']}"] = v["temporal_interp"]
            if len(v["variable_enhancers"]) > 0:
                enhance[k] = v["variable_enhancers"]

        summary["_WHITTAKER_"] = whittaker
        summary["_ENHANCE_"] = enhance
        summary["_EXAMPLE_"] = example

        self.summary = summary

class Project():

    def __init__(self, project_folder, bb, period, configuration = None):

        assert bb[::2][0] < bb[::2][1], "Invalid Bounding-Box"
        self.lonlim = bb[::2]
        assert bb[1::2][0] < bb[1::2][1], "Invalid Bounding-Box"
        self.latlim = bb[1::2]
        if not os.path.isdir(project_folder):
            os.makedirs(project_folder)
        self.folder = project_folder
        self.period = adjust_timelim_dtype(period)
        self.configuration = configuration
        self.dss = None
        self.bb = bb

        os.environ["pyWaPOR_bb"] = str(self.bb)
        os.environ["pyWaPOR_period"] = str(self.period)

        warnings.filterwarnings("ignore", message="invalid value encountered in power")

        adjust_logger(True, self.folder, "INFO")

        if os.path.isfile(os.path.join(self.folder, "se_root_in.nc")):
            self.se_root_in = open_ds(os.path.join(self.folder, "se_root_in.nc"))
        else:
            self.se_root_in = None

        se_root_outs = glob.glob(os.path.join(self.folder, "se_root_out*.nc"))
        if se_root_outs:
            self.se_root_out = open_ds(max(se_root_outs, key=os.path.getmtime))
        else:
            self.se_root_out = None

        if os.path.isfile(os.path.join(self.folder, "et_look_in.nc")):
            self.et_look_in = open_ds(os.path.join(self.folder, "et_look_in.nc"))
        else:
            self.et_look_in = None

        et_look_outs = glob.glob(os.path.join(self.folder, "et_look_out*.nc"))
        if et_look_outs:
            self.et_look_out = open_ds(max(et_look_outs, key=os.path.getmtime))
        else:
            self.et_look_out = None

        log.info("> PROJECT").add()
        log.info(self.__repr__())
        self.check_pywapor_version()
        self.check_gdal_drivers()
        log.sub().info("< PROJECT")

    def check_gdal_drivers(self):
        required_drivers = [
            "GTiff",
            "JP2OpenJPEG",
            "NETCDF",
            "HDF5",
        ]
        log.info(f"--> GDAL ({gdal.__version__}):").add()
        drivers_check = True
        for driver in required_drivers:
            if isinstance(gdal.GetDriverByName(driver), type(None)):
                log.warning(f"> Driver `{driver}` not found.")
                drivers_check = False
        
        if drivers_check:
            log.info("> All required GDAL drivers found.")
        log.sub()

    def check_pywapor_version(self):
        current_version = pywapor.__version__
        package = 'pywapor'
        log.info(f"--> pyWaPOR ({current_version}):").add()
        try:
            response = requests.get(f'https://pypi.org/pypi/{package}/json')
            response.raise_for_status()
        except Exception as _:
            log.warning("> Unable to check for pyWaPOR updates.")
        else:
            latest_version = response.json()['info']['version']
            if latest_version == current_version:
                log.info("> Up to date.")
            else:
                log.warning(f"> Latest version is '{latest_version}'.")
                log.warning("> Please update pywapor.")
        log.sub()

    def __repr__(self):
        project_str = f"""--> Project Folder:
        > {self.folder}
    --> Period:
        > {self.period[0]} - {self.period[1]}
    --> Bounding-Box:\n
                 {self.latlim[1]:8.4f}
                 ┌─────────┐
                 │         │
       {self.lonlim[0]:9.4f} │         │{self.lonlim[1]:9.4f}
                 │         │
                 └─────────┘
                 {self.latlim[0]:8.4f}\n
    --> Configuration:
        > {self.configuration}"""
        return project_str

    def load_configuration(self, name = None, summary = None, json = None):
        if not isinstance(name, type(None)) and not isinstance(summary, type(None)):
            raise ValueError("Only one of `level` and `summary` can be specified.")
        log.info("> CONFIGURATION").add()
        if not isinstance(name, type(None)):
            self.configuration = Configuration.from_name(name)
        elif not isinstance(summary, type(None)):
            summary = summary.copy()
            self.configuration = Configuration.from_summary(summary)
        elif not isinstance(json, type(None)):
            self.configuration = Configuration.from_json(json)
        else:
            raise ValueError("At least one of `level` or `summary` needs to be specified.")
        
        log.info("--> Summary:").add()
        summary_str = self.configuration.__repr__()

        for line in summary_str.split("\n"):
            log.info(line)

        log.sub().sub().info("< CONFIGURATION")
        return self.configuration
    
    def validate_project_folder(self):
        log.info("--> Checking project folder.").add()
        if not os.path.isabs(self.folder):
            self.folder = os.path.abspath(self.folder)
            log.warning(f"--> Consider specifying an absolute path to your project folder, continueing with `{self.folder}`.")
        if not os.path.isdir(self.folder):
            log.info(f"--> Folder does not exist, creating a new folder at `{self.folder}`.")
            os.makedirs(self.folder)
        dir_list = os.listdir(self.folder)
        bb = [self.lonlim[0], self.latlim[0], self.lonlim[1], self.latlim[1]]
        check = True

        if "log.txt" in dir_list:
            log_file = os.path.join(self.folder, "log.txt")
            project_folders, periods, bbs, temp_files = Project.parse_log_file(log_file)

            if len(set(project_folders)) > 1:
                log.info("--> Project folder has previously been moved.")
            else:
                ... # all good.

            if len(set(bbs)) > 1:
                log.warning("--> You've previously defined different bounding-boxes inside this project folder, this can result in unexpected behaviour.")
                check = False
            elif len(set(bbs)) == 1:
                bb_consistent = np.all([np.isclose(x, y) for x,y in zip(bbs[0], bb)])
                if not bb_consistent:
                    log.warning("--> You've changed the bounding-box of your project without changing the project folder, this can result in unexpected behaviour.")
                    check = False
            else:
                ... # all good.

            period_ = tuple([np.datetime64(x) for x in self.period])
            if len(set(periods)) > 1:
                log.warning("--> You've previously defined different periods inside this project folder, this can result in unexpected behaviour.")
                check = False
            elif len(set(periods)) == 1:
                period_consistent = period_ == periods[0]
                if not period_consistent:
                    log.warning("--> You've changed the period of your project without changing the project folder, this can result in unexpected behaviour.")
                    check = False
            else:
                ... # all good.

            if len(dir_list) > 1:
                self.clean_project_folder()

        elif dir_list:
            log.warning("--> Consider specifying an empty project folder.")
            check = False
        else:
            ... # all good.

        if check:
            log.info("--> All good.")

        log.sub()

    def clean_project_folder(self):

        n_corrupt = n_wrong = n_unkown = n_good = n_temp = 0
        to_remove = list()
        fhs = glob.glob(os.path.join(self.folder, "**/*.nc"), recursive=True)
        log_file = os.path.join(self.folder, "log.txt")

        if os.path.isfile(log_file):
            temp_files = Project.parse_log_file(log_file)[-1]
            for fh in temp_files:
                if fh in fhs:
                    fhs.remove(fh)
                if os.path.isfile(fh):
                    to_remove.append(fh)
                    n_temp += 1

        log.info(f"--> Checking {len(fhs)} files.").add()

        for fh in fhs.copy():
            corrupt = is_corrupt_or_empty(fh)
            if corrupt:
                to_remove.append(fh)
                fhs.remove(fh)
                n_corrupt += 1
            
        for fh in fhs:
            wrong = has_wrong_bb_or_period(fh, self.bb, self.period)
            if wrong:
                to_remove.append(fh)
                n_wrong += 1
            elif isinstance(wrong, type(None)):
                n_unkown += 1
            else:
                n_good += 1

        if n_temp > 0:
            log.info(f"--> Found {n_temp} temporary files that can be removed.")
        log.info(f"--> Found {n_good} files in good condition.")
        if n_corrupt > 0:
            log.warning(f"--> Found {n_corrupt} corrupt or empty files.")
        if n_wrong > 0:
            log.warning(f"--> Found {n_wrong} files with an incorrect `bb` or `period`.")
        if n_unkown > 0:
            log.info(f"--> Couldn't determine state of {n_unkown} files.").sub()
        
        if n_corrupt + n_wrong + n_temp > 0:
            response = input(prompt = f"Remove {n_corrupt} corrupt, {n_wrong} wrong and {n_temp} temporary files? (Yes/No)")
            if response in ["Y", "y", "Yes", "yes"]:
                for x in to_remove:
                    remove_ds(x)

    @staticmethod
    def parse_log_file(log_file):

        with open(log_file, "r", encoding='utf8') as f:
            log_string = f.read()

        project_folders = re.findall(r"--> Project Folder:\n        > (.*)", log_string)
        
        periods_ = re.findall(r"--> Period:\n        > (.*) - (.*)", log_string)
        periods = [tuple([np.datetime64(x) for x in period]) for period in periods_]

        bbs_ = re.findall(r"--> Bounding-Box:\n\n\s*(\d{1,2}.\d{4})\n.*\n.*\n\s*(\d{1,2}.\d{4})\D*(\d{1,2}.\d{4})\n.*\n.*\n\D*(\d{1,2}.\d{4})", log_string)
        bbs = [tuple([[float(x) for x in bb][i] for i in [1,3,2,0]]) for bb in bbs_]
        
        temp_files = re.findall(r"--> Unable to delete temporary file `(.*)`", log_string)

        return project_folders, periods, bbs, temp_files

    def set_passwords(self):

        log.info("> PASSWORDS").add()

        req_accounts = {
            "MODIS": "NASA",
            "SRTM": "NASA",
            "MERRA2": "NASA",
            "TERRA": "TERRA",
            "ERA5": "CDS",
            "LANDSAT": "EARTHEXPLORER",
            "SENTINEL2": "COPERNICUS_DATA_SPACE",
            "SENTINEL3": "COPERNICUS_DATA_SPACE",
        }

        all_accounts = list()
        for v in self.configuration.summary.values():
            if isinstance(v, set):
                for source_product in v:
                    name = Configuration.source_func(source_product)
                    account = req_accounts.get(name, None)
                    if not isinstance(account, type(None)):
                        all_accounts.append(account)

        x = set(all_accounts)
        log.info(f"--> Accounts needed for `{'`, `'.join(x)}`.").add()

        for account in x:
            _ = pywapor.collect.accounts.get(account)

        log.sub().info("--> All set!")
        log.sub().info("< PASSWORDS")

    def download_data(self, buffer_timelim = True):
        assert not isinstance(self.configuration, type(None)), "Please load a configuration before continueing."

        log.info("> DOWNLOADER").add()

        example_t_vars = [x for x in ["lst", "bt"] if x in self.configuration.full.keys()]
        example_sources = {k:v for k,v in self.configuration.full.items() if k in example_t_vars}
        other_sources = {k:v for k,v in self.configuration.full.items() if k not in example_t_vars}

        if buffer_timelim:
            bins = time_bins(self.period, 1)
            adjusted_timelim = [bins[0], bins[-1]]
            buffered_timelim = [adjusted_timelim[0] - np.timedelta64(3, "D"), 
                                adjusted_timelim[1] + np.timedelta64(3, "D")]
        else:
            adjusted_timelim = self.period
            buffered_timelim = self.period

        example_dss, _ = collect_sources(
                            self.folder,
                            example_sources,
                            self.latlim,
                            self.lonlim,
                            adjusted_timelim,
                            landsat_order_only = True
                            )
        
        other_dss, _ = collect_sources(
                            self.folder,
                            other_sources,
                            self.latlim,
                            self.lonlim,
                            buffered_timelim,
                            )
        
        # If there are example-t variables that rely on landsat, try one more time to collect them.
        if np.any(list({var: np.any([product_info["source"] == "LANDSAT" for product_info in info["products"]]) for var, info in example_sources.items()}.values())):
            example_dss, _ = collect_sources(
                                            self.folder, 
                                            example_sources, 
                                            self.latlim, 
                                            self.lonlim, 
                                            adjusted_timelim
                                            )

        self.dss = {**example_dss, **other_dss}
        
        log.sub().info("< DOWNLOADER")
    
        return self.dss
    
    def run_pre_se_root(self, forced = False):
        if isinstance(self.se_root_in, type(None)) or forced:
            self.se_root_in = pywapor.pre_se_root.main(self.folder, self.latlim, self.lonlim, self.period, sources = self.configuration.se_root)
        else:
            log.info("> PRE_SE_ROOT").add()
            log.info(f"--> Re-using `{os.path.split(self.se_root_in.encoding['source'])[-1]}`.")
            log.sub().info("< PRE_SE_ROOT")
        return self.se_root_in

    def run_se_root(self, se_root_version = "v3", export_vars = "default", chunks = {"time": -1, "x": 500, "y": 500}):
        self.se_root_out = pywapor.se_root.main(
                                                self.se_root_in, 
                                                se_root_version = se_root_version, 
                                                export_vars = export_vars, 
                                                chunks = chunks
                                                )
        return self.se_root_out
    
    def run_pre_et_look(self, forced = False):
        if isinstance(self.et_look_in, type(None)) or forced:
            self.et_look_in = pywapor.pre_et_look.main(self.folder, self.latlim, self.lonlim, self.period, sources = self.configuration.et_look)
        else:
            log.info("> PRE_ET_LOOK").add()
            log.info(f"--> Re-using `{os.path.split(self.et_look_in.encoding['source'])[-1]}`.")
            log.sub().info("< PRE_ET_LOOK")
        return self.et_look_in

    def run_et_look(self, et_look_version = "v3", export_vars = "default", chunks = {"time_bins": -1, "x": 500, "y": 500}):
        self.et_look_out = pywapor.et_look.main(
                                                self.et_look_in, 
                                                et_look_version = et_look_version, 
                                                export_vars = export_vars, 
                                                chunks = chunks
                                                )
        return self.et_look_out

if __name__ == "__main__":

    timelim = ["2023-11-01", "2023-11-30"]
    latlim = [21.9692194682626933, 21.9939120838340507]
    lonlim = [91.9371349243682801, 91.9657566608824339]
    bb = [91.9371349243682801, 21.9692194682626933, 91.9657566608824339, 21.9939120838340507]
    project_folder = r"/Users/hmcoerver/Local/pywapor_bgdX"

    adjust_logger(True, project_folder, "INFO")

    all_configs = {
        "WaPOR2_level_1",
        "WaPOR2_level_2",
        "WaPOR2_level_3",
        "WaPOR2_level_1",
        "WaPOR2_level_2",
        "WaPOR2_level_3",
        "WaPOR3_level_2",
        "WaPOR3_level_3",
        "nrt",
        "all_in",
    }

    project = pywapor.Project(project_folder, bb, timelim)

    # project.validate_project_folder()

    # project.load_configuration(name = "WaPOR3_level_2")

    summary = {
        # Define which products to use.
        'elevation': {},
        'meteorological': {},
        'optical': {'LANDSAT.LC09_SR'},
        'precipitation': {},
        'solar radiation': {},
        'statics': {},
        'thermal': {},
        'soil moisture': {},

        # Define which product to reproject the other products to.
        '_EXAMPLE_': 'LANDSAT.LC09_SR', 

        # # Define any special functions to apply to a specific variable.
        # '_ENHANCE_': {"bt": ["pywapor.enhancers.dms.thermal_sharpener.sharpen"],},

        # # Choose which products should be gapfilled.
        # '_WHITTAKER_': {
        #     'SENTINEL2.S2MSI2A_R20m': {'lmbdas': 1000.0, 'method': 'whittaker'}, 
        #     'VIIRSL1.VNP02IMG': {'a': 0.85, 'lmbdas': 1000.0, 'method': 'whittaker'}},
        }
    
    project.load_configuration(summary = summary)



    project.set_passwords()
    dss = project.download_data()
    
    # se_root_in = project.run_pre_se_root()
    # se_root = project.run_se_root()
    # et_look_in = project.run_pre_et_look()
    # et_look = project.run_et_look()





