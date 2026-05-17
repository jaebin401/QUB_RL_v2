import isaacgym
from isaacgym import gymapi

def check_physics_properties():
    # 1. Gym 엔진 초기화
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

    # 2. 에셋(URDF) 경로 설정
    asset_root = "/home/qub1/Jaebin/QUB_RL_v2/resources/robots/QUB/urdf"
    asset_file = "QUB.urdf"
    
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.armature = 0.01

    print(f"Loading {asset_file}...")
    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
    
    # 3. 가상의 환경(Env)과 로봇(Actor) 생성 (API 요구사항)
    env = gym.create_env(sim, gymapi.Vec3(-1, -1, -1), gymapi.Vec3(1, 1, 1), 1)
    pose = gymapi.Transform()
    actor = gym.create_actor(env, asset, pose, "QUB_robot", 0, 1)

    # 4. 생성된 로봇(Actor)으로부터 링크 물성치 추출
    body_names = gym.get_actor_rigid_body_names(env, actor)
    body_props = gym.get_actor_rigid_body_properties(env, actor)

    total_mass = 0.0
    print("\n" + "="*85)
    print(f"{'Rigid Body Name':<25} | {'Mass (kg)':<10} | {'Ixx':<10} | {'Iyy':<10} | {'Izz':<10}")
    print("-" * 85)
    
    for i, prop in enumerate(body_props):
        name = body_names[i]
        mass = prop.mass
        ixx = prop.inertia.x.x
        iyy = prop.inertia.y.y
        izz = prop.inertia.z.z
        total_mass += mass
        
        # 위험 경고 (질량이 너무 작거나 관성 모멘트가 1e-6 미만인 경우)
        warning = ""
        if mass < 0.01 or ixx < 1e-6 or iyy < 1e-6 or izz < 1e-6:
            warning = "  <-- ⚠️ WARNING: 수치 발산 위험!"
            
        print(f"{name:<25} | {mass:<10.4f} | {ixx:<10.6f} | {iyy:<10.6f} | {izz:<10.6f}{warning}")

    print("-" * 85)
    print(f"Total Robot Mass: {total_mass:.4f} kg")
    print("="*85 + "\n")

    gym.destroy_sim(sim)

if __name__ == "__main__":
    check_physics_properties()