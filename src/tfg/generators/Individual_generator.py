def generar_individuo(random, args):
    print(args['users'])
    num_users = len(args['users'])
    num_platforms = len(args['platforms_indexed'])
    individuo = [[random.randint(1, num_platforms) for _ in range(12)] for _ in range(num_users)]
    return individuo

def get_platform_name(platform_id, platforms_indexed):
    return platforms_indexed.get(str(platform_id))