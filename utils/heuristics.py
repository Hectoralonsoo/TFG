def encontrar_combinacion_optima(num_usuarios, planes_compactos):
    """
    Calcula la combinación óptima de planes para cubrir el número de usuarios dado.
    planes_compactos = [(perfiles, precio), ...]
    """
    dp = [(float('inf'), []) for _ in range(num_usuarios + 1)]
    dp[0] = (0, [])

    for i in range(1, num_usuarios + 1):
        for perfiles, precio in planes_compactos:
            if i <= perfiles:
                if precio < dp[i][0]:
                    dp[i] = (precio, [(1, perfiles, precio)])
            else:
                prev_cost, prev_combo = dp[i - perfiles]
                if prev_cost + precio < dp[i][0]:
                    nueva_combo = prev_combo.copy()
                    encontrado = False
                    for idx, (cant, perf, prec) in enumerate(nueva_combo):
                        if perf == perfiles and prec == precio:
                            nueva_combo[idx] = (cant + 1, perf, prec)
                            encontrado = True
                            break
                    if not encontrado:
                        nueva_combo.append((1, perfiles, precio))
                    dp[i] = (prev_cost + precio, nueva_combo)

    return dp[num_usuarios]
