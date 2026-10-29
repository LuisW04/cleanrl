import numpy as np
import os
#for metadata:
import json
from ocatari.ram.extract_ram_info import get_max_objects, get_class_dict
from ocatari.ram.game_objects import NoObject, ValueObject

#ms pacman: num_objects=14 -> player, 4 ghosts, fruit, 4 powerpills, num_pills, fruit, score, life

class DatasetCollector: 
    def __init__(self, buffer_size, num_envs, obs_space, action_space, num_objects=14, obj_feature_dim=5):
        #self.folder_path = folder_path #folder_path, exp_name, 
        #self.exp_name = exp_name
        self.buffer_size = buffer_size
        self.num_objects = num_objects
        self.obj_feature_dim = obj_feature_dim
        self.ptr = 0
        self.dataset_chunk_counter = 0
        self.observations = np.zeros((buffer_size, num_envs)+obs_space.shape, dtype=obs_space.dtype)
        self.actions = np.zeros((buffer_size, num_envs)+action_space.shape, dtype=action_space.dtype)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.terminations = np.zeros((buffer_size, num_envs), dtype=np.bool_)
        self.truncations = np.zeros((buffer_size, num_envs), dtype=np.bool_)
        self.objects = np.zeros((buffer_size, num_envs, num_objects, obj_feature_dim), dtype=np.int16)

    def encode_ms_pacman_objects(self, objects_per_env):
        encoded = np.zeros((len(objects_per_env), self.num_objects, self.obj_feature_dim), dtype=np.int16)
        for env_idx, object_list in enumerate(objects_per_env):
            for obj_idx, obj in enumerate(object_list[:10]): #10 = player + 4 ghosts + fruit + 4 powerpills
                (x, y), (w, h) = obj.xy, obj.wh
                encoded[env_idx][obj_idx] = [x, y, w, h]
            collected_pills = 0
            uncollected_pills = 0
            for obj_idx, obj in enumerate(object_list[10:-3]):
                (w, h) = obj.wh
                if w == 0 and h == 0:
                    collected_pills += 1
                else:
                    uncollected_pills +=1
            encoded[env_idx][10] = [collected_pills,uncollected_pills,0,0]
            #print("collected: ", collected_pills, "uncollected: ", uncollected_pills)
            for obj_idx, obj in enumerate(object_list[-3:]): #-3 = fruit + score + life
                (x, y), (w, h) = obj.xy, obj.wh
                encoded[env_idx][obj_idx + 11] = [x, y, w, h]
        return encoded

    def encode_objects(self, objects_per_env):
        encoded = np.zeros((len(objects_per_env), self.num_objects, self.obj_feature_dim), dtype = np.int32)
        for env_idx, obj_list in enumerate(objects_per_env):
            for obj_idx, obj in enumerate(obj_list):
                (x, y, w, h) = obj.xywh
                encoded[env_idx][obj_idx] = [x, y, w, h, getattr(obj, "value", 0)] #perhaps -1 or NaN is better because value can be 0
        return encoded

    def add(self, obs, action, reward, terminated, truncated, objects_per_env):
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminations[self.ptr] = terminated
        self.truncations[self.ptr] = truncated
        self.objects[self.ptr] = self.encode_objects(objects_per_env)
        self.ptr += 1

    @property
    def full(self):
        return self.ptr >= self.buffer_size

    def save(self, folder_path, exp_name):
        file_name = f"{exp_name}_dataset_{self.dataset_chunk_counter}"
        os.makedirs(folder_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(folder_path, file_name),
            observations = self.observations[:self.ptr],
            actions = self.actions[:self.ptr],
            rewards = self.rewards[:self.ptr],
            terminations = self.terminations[:self.ptr],
            truncations = self.truncations[:self.ptr],
            objects = self.objects[:self.ptr]
        )
        self.ptr = 0
        self.dataset_chunk_counter += 1

    def create_metadata(self, oc_env, folder_path, exp_name):
        path = os.path.join(folder_path, f"{exp_name}_metadata.json")
        game = oc_env.game_name
        table = get_max_objects(game, oc_env.hud)
        classes = get_class_dict(game)
        table_no_hud = get_max_objects(game, False)
        slots = [{
            "category": category,
            "is_hud": category not in table_no_hud,
            "is_value": issubclass(classes[category], ValueObject)
        } for category, n in table.items() for _ in range(n)]
        
        #check if data is correct
        objects = oc_env.objects
        assert len(slots) == len(objects) == self.num_objects, (len(slots), len(objects), self.num_objects)
        for i, o in enumerate(objects):
            if not isinstance(o, NoObject):
                assert o.category == slots[i]["category"], (i, o.category, slots[i]["category"])
        
        meta_data = {
            "num_slots": len(slots),
            "slots": slots,
            "features": ["x", "y", "w", "h", "value"], #could add existence flag
            "action_meanings": oc_env.get_action_meanings(),
            "env": {"id": game, "mode": oc_env.mode, "hud": oc_env.hud, "obs_mode": oc_env.obs_mode}
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        

