# https://github.com/skfoo/ComfyUI-Coziness

import folder_paths
import comfy.utils
import comfy.sd
import os
import re

class MultiLoraLoader:
    def __init__(self):
        self.selected_loras = SelectedLoras()
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "clip": ("CLIP", ),
                              "text": ("STRING", {
                                "multiline": True,
                                "default": ""}),
                            }}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_loras"
    CATEGORY = "loaders"

    def load_loras(self, model, clip, text):
        result = (model, clip)
        
        lora_items = self.selected_loras.updated_lora_items_with_text(text)

        if len(lora_items) > 0:
            for item in lora_items:
                result = item.apply_lora(result[0], result[1])
            
        return result
 
# maintains a list of lora objects made from a prompt, preserving loaded loras across changes
class SelectedLoras:
    def __init__(self):
        self.lora_items = []

    # returns a list of loaded loras using text from LoraTextExtractor
    def updated_lora_items_with_text(self, text):
        available_loras = self.available_loras()
        self.update_current_lora_items_with_new_items(self.items_from_lora_text_with_available_loras(text, available_loras))
        
        for item in self.lora_items:
            if item.lora_name not in available_loras:
                raise ValueError(f"Unable to find lora with name '{item.lora_name}'")
            
        return self.lora_items

    def available_loras(self):
        return folder_paths.get_filename_list("loras")
    
    def items_from_lora_text_with_available_loras(self, lora_text, available_loras):
        return LoraItemsParser.parse_lora_items_from_text(lora_text, self.dictionary_with_short_names_for_loras(available_loras))
    
    def dictionary_with_short_names_for_loras(self, available_loras):
        result = {}
        
        for path in available_loras:
            result[os.path.splitext(os.path.basename(path))[0]] = path
        
        return result

    def update_current_lora_items_with_new_items(self, lora_items):
        if self.lora_items != lora_items:
            existing_by_name = dict([(existing_item.lora_name, existing_item) for existing_item in self.lora_items])
            
            for new_item in lora_items:
                new_item.move_resources_from(existing_by_name)
            
            self.lora_items = lora_items

class LoraItemsParser:

    @classmethod
    def parse_lora_items_from_text(cls, lora_text, loras_by_short_names = {}, default_weight=1, weight_separator=":"):
        return cls(lora_text, loras_by_short_names, default_weight, weight_separator).execute()

    def __init__(self, lora_text, loras_by_short_names, default_weight, weight_separator):
        self.lora_text = lora_text
        self.loras_by_short_names = loras_by_short_names
        self.default_weight = default_weight
        self.weight_separator = weight_separator
        self.prefix_trim_re = re.compile("\A<(lora|lyco):")
        self.comment_trim_re = re.compile("\s*#.*\Z")
    
    def execute(self):

        out_loras = []

        for line in self.lora_text.splitlines():
            description = self.description_from_line(line)
            name, model_weight, clip_weight, block_type = self.parse_lora_description(description)
            if name is not None:
                out_loras.append(LoraItem(name, model_weight, clip_weight, block_type))
                    
        return out_loras
    
    def parse_lora_description(self, description):
        if description is None:
            return (None, None, None, None)
        
        lora_name = None
        strength_model = self.default_weight
        strength_clip = None
        block_type = "blocks_all"
        
        parts = description.split(self.weight_separator)
    
        try:
            if len(parts) == 1:  # Only lora name
                lora_name = parts[0]
            elif len(parts) == 2:  # lora name and model weight
                lora_name, last_param = parts
                if last_param.startswith("blocks_"):
                    block_type = last_param
                else:
                    strength_model = float(last_param)
            elif len(parts) == 3:  # lora name, model weight, and either clip weight or block type
                lora_name, strength_model, last_param = parts
                strength_model = float(strength_model)
                if last_param.startswith("blocks_"):
                    block_type = last_param
                else:
                    strength_clip = float(last_param)
            elif len(parts) == 4:  # lora name, model weight, clip weight, and block type
                lora_name, strength_model, strength_clip, block_type = parts
                strength_model = float(strength_model)
                strength_clip = float(strength_clip)
        except ValueError as e:
            raise ValueError(f"Invalid description format: {description}") from e
        
        if strength_clip is None:
            strength_clip = strength_model

        return (self.loras_by_short_names.get(lora_name, lora_name), strength_model, strength_clip, block_type)


    def description_from_line(self, line):
        result = self.comment_trim_re.sub("", line.strip())
        result = self.prefix_trim_re.sub("", result.removesuffix(">"))
        return result if len(result) > 0 else None
        

class LoraItem:
    def __init__(self, lora_name, strength_model, strength_clip, blocks_type):
        self.lora_name = lora_name
        self.strength_model = strength_model
        self.strength_clip = strength_clip
        self.blocks_type = blocks_type
        self._loaded_lora = None
    
    def __eq__(self, other):
        return self.lora_name == other.lora_name and self.strength_model == other.strength_model and self.strength_clip == other.strength_clip and self.blocks_type == other.blocks_type
    
    def get_lora_path(self):
        return folder_paths.get_full_path("loras", self.lora_name)
        
    def move_resources_from(self, lora_items_by_name):
        existing = lora_items_by_name.get(self.lora_name)
        if existing is not None:
            self._loaded_lora = existing._loaded_lora
            existing._loaded_lora = None

    def apply_lora(self, model, clip):
        if self.is_noop:
            return (model, clip)

        filtered_lora = self.get_filtered_lora()
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, filtered_lora, self.strength_model, self.strength_clip)
        return (model_lora, clip_lora)

    def make_base_lora_key(self, key: str) -> str:
        prefixes = ["diffusion_model.", "transformer."]
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
                
        return key
    
    def get_filtered_lora(self):
        if "blocks_all" in self.blocks_type:
            return self.lora_object
            
        filtered_lora = {}
        for key, value in self.lora_object.items():
            base_key = self.make_base_lora_key(key)
            
            if "blocks_single" in self.blocks_type and "single_blocks" in base_key:
                filtered_lora[key] = value
            elif "blocks_double" in self.blocks_type and "double_blocks" in base_key:
                filtered_lora[key] = value
                
        return filtered_lora

    @property
    def lora_object(self):
        if self._loaded_lora is None:
            lora_path = self.get_lora_path()
            if lora_path is None:
                raise ValueError(f"Unable to get file path for lora with name '{self.lora_name}'")
            self._loaded_lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        return self._loaded_lora

    @property
    def is_noop(self):
        return self.strength_model == 0 and self.strength_clip == 0

class LoraTextExtractor:
    def __init__(self):
        self.lora_spec_re = re.compile("(<(?:lora|lyco):[^>]+>)")
        self.selected_loras = SelectedLoras()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "text": ("STRING", {
                                "multiline": True,
                                "default": ""}),
                            }}

    RETURN_TYPES = ("STRING", "STRING", "LORA_STACK")
    RETURN_NAMES = ("Filtered Text", "Extracted Loras", "Lora Stack")
    FUNCTION = "process_text"
    CATEGORY = "utils"

    def process_text(self, text):
        extracted_loras = "\n".join(self.lora_spec_re.findall(text))
        filtered_text = self.lora_spec_re.sub("", text)

        # the stack format is a list of tuples of full path, model weight, clip weight,
        # e.g. [('styles\\abstract.safetensors', 0.8, 0.8)]
        lora_stack = [(item.get_lora_path(), item.strength_model, item.strength_clip) for item in self.selected_loras.updated_lora_items_with_text(extracted_loras)]
        
        return (filtered_text, extracted_loras, lora_stack)

NODE_CLASS_MAPPINGS = {
    "MultiLoraLoader-70bf3d77": MultiLoraLoader,
    "LoraTextExtractor-b1f83aa2": LoraTextExtractor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiLoraLoader-70bf3d77": "MultiLora Loader",
    "LoraTextExtractor-b1f83aa2": "Lora Text Extractor",
}
