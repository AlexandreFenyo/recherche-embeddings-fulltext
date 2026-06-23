def collection_name(model_name: str, strategy_name: str) -> str:
    """Nom de collection Qdrant dérivé du modèle et de la stratégie.

    Source unique de vérité partagée par l'indexation et la recherche : toute
    divergence casserait silencieusement la correspondance index <-> requête.
    Pas de troncature : les noms restent sous la limite du système de fichiers
    (< 255 octets) tout en évitant les collisions par coupure.
    """
    safe = (
        strategy_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(",", "")
    )
    return f"{model_name}__{safe}"
